import os
from tqdm import tqdm
from math import log, sqrt, pi

import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from PIL import Image
import torch.utils.data as data

from model import Glow

device = torch.device("cuda")

parser = argparse.ArgumentParser(description="Glow trainer")
parser.add_argument("--batch", default=16, type=int, help="batch size")
parser.add_argument("--num_threads", default=4, type=int, help="threads for loading data")
parser.add_argument("--n_epoch", default=50, type=int, help="maximum epochs")
parser.add_argument("--epoch_count", type=int, default=1, help="the starting epoch count")
parser.add_argument("--iter", default=200000, type=int, help="maximum iterations")
parser.add_argument("--save_epoch_freq", type=int, default=10, help="frequency of saving checkpoints and samples at the end of epochs")
parser.add_argument(
    "--n_flow", default=32, type=int, help="number of flows in each block"
)
parser.add_argument("--n_block", default=4, type=int, help="number of blocks")
parser.add_argument(
    "--no_lu",
    action="store_true",
    help="use plain convolution instead of LU decomposed version",
)
parser.add_argument(
    "--affine", action="store_true", help="use affine coupling instead of additive"
)
parser.add_argument("--sigmoid", action="store_true", help="use sigmoid in affine coupling to stabilize training")
parser.add_argument("--n_bits", default=5, type=int, help="number of bits")
parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
parser.add_argument("--img_size", default=64, type=int, help="image size")
parser.add_argument("--img_channel", default=3, type=int, help="image channel")
parser.add_argument("--temp", default=0.7, type=float, help="temperature of sampling")
parser.add_argument("--n_sample", default=20, type=int, help="number of samples")
parser.add_argument("path", metavar="PATH", type=str, help="Path to image directory")


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir) or os.path.islink(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:len(images)]


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)


def get_train_transforms(image_size):
    transform_list = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]
    return transforms.Compose(transform_list)


def calc_z_shapes(n_channel, input_size,  n_block):
    z_shapes = []

    for i in range(n_block - 1):
        # except the last block, each block halves H and W and doubles C
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))  # the last block halves H and W and quadruples C

    return z_shapes


def calc_loss(log_p, logdet, image_size, n_bins, image_channel=3):
    """
    calculate NLL bit per dimension(bits/dim)
    """
    n_pixel = image_size * image_size * image_channel

    loss = -log(n_bins) * n_pixel  # the log-likelihood of added noise
    loss = loss + logdet + log_p

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )


def train(args, model, optimizer):
    # print and save args
    print(args)
    argsDict = args.__dict__
    with open('checkpoint/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

    train_dataset = ImageFolder(args.path, get_train_transforms(args.img_size))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.num_threads, drop_last=True)
    n_bins = 2.0 ** args.n_bits

    z_sample = []
    z_shapes = calc_z_shapes(args.img_channel, args.img_size, args.n_block)
    for z in z_shapes:
        z_new = torch.randn(args.n_sample, *z) * args.temp
        z_sample.append(z_new.to(device))
    total_iters = 0  # the total number of training iterations
    with tqdm(range(args.epoch_count, args.n_epoch + 1)) as pbar:
        for epoch in pbar:
            epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
            for i, image in enumerate(train_dataloader):  # inner loop within one epoch
                image = image.to(device)

                image = image * 255

                if args.n_bits < 8:
                    image = torch.floor(image / 2 ** (8 - args.n_bits))

                image = image / n_bins - 0.5
                total_iters += args.batch
                epoch_iter += args.batch
                if epoch == 1 and i == 0:
                    with torch.no_grad():
                        log_p, logdet, _ = model.module(image + torch.rand_like(image) / n_bins)
                        continue
                else:
                    log_p, logdet, _ = model(image + torch.rand_like(image) / n_bins)

                    logdet = logdet.mean()

                    loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins, args.img_channel)
                    optimizer.zero_grad()
                    loss.backward()
                    # warmup_lr = args.lr * min(1, i * batch_size / (50000 * 10))
                    warmup_lr = args.lr
                    optimizer.param_groups[0]["lr"] = warmup_lr
                    optimizer.step()

                    pbar.set_description(f"Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; lr: {warmup_lr:.7f}")

            if epoch % args.save_epoch_freq == 0:
                with torch.no_grad():
                    utils.save_image(
                        model_single.reverse(z_sample).cpu().data,
                        f"sample/{str(epoch).zfill(6)}.png",
                        normalize=True,
                        nrow=10,
                        range=(-0.5, 0.5),
                    )
                torch.save(model.state_dict(), f"checkpoint/model_{str(epoch).zfill(6)}.pt")
                torch.save(optimizer.state_dict(), f"checkpoint/optim_{str(epoch).zfill(6)}.pt")


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    model_single = Glow(3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu, use_sigmoid=args.sigmoid)
    model = nn.DataParallel(model_single)
    # model = model_single
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(args, model, optimizer)
