import argparse
import os

# Default --data_root per dataset, matching the repo layout: data/cub/data/waterbird_complete95_forest2water2/
# and data/celeba/. Resolved from this file, so it does not depend on the working directory.
REPO_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')
DEFAULT_DATA_ROOTS = {
    'cub': os.path.join(REPO_DATA_DIR, 'cub', 'data'),
    'celeba': REPO_DATA_DIR,
}

parser = argparse.ArgumentParser()

parser.add_argument('--name', default='temp', type=str)

# dataset configuration
parser.add_argument('--dataset', default='celeba', type=str, help='dataset celeba[default]')
parser.add_argument('--data_root', default=None, type=str, help='dataset root dir (default: the repo data/ layout, see DEFAULT_DATA_ROOTS)')
parser.add_argument('--image_size', default=224, type=int)
parser.add_argument('--target_attr', default=9, type=int)
parser.add_argument('--bias_attr', default=20, type=int)
parser.add_argument('--pseudo_bias', default=None, type=str)
parser.add_argument('--val_frac', default=1.0, type=float, help='fraction of each CelebA validation group to keep')
parser.add_argument('--num_classes', default=2, type=int)

# model configuration
parser.add_argument('--model', default='resnet50', type=str)
parser.add_argument('--pretrained', default='imagenet', type=str)

# optimization configuration
parser.add_argument('--optimizer', default='sgd', type=str)
parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
parser.add_argument('--batch_size', '--batch-size', default=64, type=int, help='mini-batch size')
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int, help='print frequency (default: 10)')

parser.add_argument('--seed', type=int, default=1)


def get_arguments():
    args = parser.parse_args()
    if args.data_root is None:
        args.data_root = DEFAULT_DATA_ROOTS[args.dataset]
    return args
