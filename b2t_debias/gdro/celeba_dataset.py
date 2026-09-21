import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class CustomCelebA(Dataset):
    """
    CelebA read from the comma-separated CSVs produced by data/convert_celeba_annotations.py
    (repo root), laid out as <root>/celeba/{list_attr_celeba.csv, list_eval_partition.csv,
    img_align_celeba/data/}. Replaces torchvision.datasets.CelebA, which also requires the
    identity/bbox/landmark files and tries to download them from Google Drive.
    """
    base_folder = 'celeba'
    split_dict = {'train': 0, 'valid': 1, 'test': 2}

    def __init__(self, root, split, target_attr, bias_attr, transform, pseudo_bias=None):
        self.root = root
        self.transform = transform

        attr_df = pd.read_csv(os.path.join(root, self.base_folder, 'list_attr_celeba.csv'))
        split_df = pd.read_csv(os.path.join(root, self.base_folder, 'list_eval_partition.csv'))
        mask = (split_df['partition'] == self.split_dict[split]).values

        self.filename = attr_df['image_id'].values[mask]
        # Same layout as torchvision: 40 attribute columns in file order, -1/1 mapped to 0/1,
        # so --target_attr 9 is Blond_Hair and --bias_attr 20 is Male.
        self.attr = torch.from_numpy((attr_df.drop(columns='image_id').values[mask] + 1) // 2)

        self.targets = self.attr[:, target_attr]
        if pseudo_bias is not None:
            self.biases = torch.load(pseudo_bias)
        else:
            self.biases = self.attr[:, bias_attr]

    def __len__(self):
        return len(self.filename)

    def __getitem__(self, index):
        X = Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", "data", self.filename[index]))
        y = self.targets[index]
        a = self.biases[index]

        if self.transform is not None:
            X = self.transform(X)

        ret_obj = {'x': X,
                   'y': y,
                   'a': a,
                   'dataset_index': index,
                   'filename': self.filename[index],
                   }

        return ret_obj
