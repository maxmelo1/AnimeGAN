import argparse

from train_wgan import train as train_wgan
from train_wgan_gp import train as train_wgangp

parser = argparse.ArgumentParser(description='Train Anime WGAN.')
parser.add_argument('--type', type=str, help='WGAN or WGAN-GP', choices=['WGAN', 'WGANGP'])
parser.add_argument('--epochs', type=int, help='N of epochs')
parser.add_argument('--bs', type=int, help='Batch size', default=256)

args = parser.parse_args()

type = args.type

if type == 'WGAN':
    train_wgan(args.epochs, bs=args.bs)
else:
    train_wgangp(args.epochs, bs=args.bs)