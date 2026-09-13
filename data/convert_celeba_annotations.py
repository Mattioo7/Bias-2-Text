"""
Convert the official CelebA annotation .txt files into the .csv layout that
data/celeba.py expects (pd.read_csv with a single header row).

  list_attr_celeba.txt      -> list_attr_celeba.csv       (image_id + 40 attributes)
  list_eval_partition.txt   -> list_eval_partition.csv    (image_id, partition)

Run from the repo root:  uv run python data/convert_celeba_annotations.py
"""
import os
import pandas as pd

CELEBA_DIR = os.path.join('data', 'celeba')


def convert_attributes(src, dst):
    # Line 1 is the image count, line 2 the 40 attribute names (no column name
    # for the filename), then whitespace-aligned rows of -1/1 values.
    with open(src) as f:
        n_images = int(f.readline().strip())
        columns = ['image_id'] + f.readline().split()
    df = pd.read_csv(src, sep=r'\s+', skiprows=2, header=None, names=columns)
    assert len(df) == n_images, f"expected {n_images} rows, got {len(df)}"
    df.to_csv(dst, index=False)
    return df


def convert_partition(src, dst):
    # No header at all: "<filename> <partition>".
    df = pd.read_csv(src, sep=r'\s+', header=None, names=['image_id', 'partition'])
    df.to_csv(dst, index=False)
    return df


if __name__ == '__main__':
    attr = convert_attributes(
        os.path.join(CELEBA_DIR, 'list_attr_celeba.txt'),
        os.path.join(CELEBA_DIR, 'list_attr_celeba.csv'))
    part = convert_partition(
        os.path.join(CELEBA_DIR, 'list_eval_partition.txt'),
        os.path.join(CELEBA_DIR, 'list_eval_partition.csv'))

    # celeba.py joins the two frames positionally, so the row order must match.
    assert attr['image_id'].equals(part['image_id']), "image_id order differs between files"

    print(f"list_attr_celeba.csv:     {len(attr)} rows, {len(attr.columns)} columns")
    print(f"list_eval_partition.csv:  {len(part)} rows")
    print("split sizes:", part['partition'].value_counts().sort_index().to_dict())
