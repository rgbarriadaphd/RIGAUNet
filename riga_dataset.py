import matplotlib.pyplot as plt
from numpy import float32, long
from torch.utils.data import Dataset
import pandas as pd
import os
import torch
from skimage import io, transform
from skimage.color import rgb2gray
import numpy as np


class RIGADataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.riga_frame = pd.read_csv(csv_file, encoding='utf8')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.riga_frame)

    def __getitem__(self, idx):
        idx -= 1
        if torch.is_tensor(idx):
            idx = idx.tolist()

        raw = io.imread(os.path.join(self.root_dir, self.riga_frame.iloc[idx, 0]))
        mask = rgb2gray(io.imread(os.path.join(self.root_dir, self.riga_frame.iloc[idx, 2])))

        sample = {'raw': raw,
                  'mask': mask,
                  }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def show_sample(self, idx):
        if not 0 < idx <= len(self):
            return

        sample = self[idx - 1]

        fig = plt.figure()
        plt.tight_layout()

        fig.add_subplot(1, 2, 1)
        plt.imshow(sample['raw'])

        fig.add_subplot(1, 2, 2)
        plt.imshow(sample['mask'])

        plt.show()


class CropFundus(object):
    """Crops the image size to adjust the image just to the fundus
       Args
        w_crop (int): Horizontal (width crop)
        h_crop (int): Vertical (height crop)
    """

    def __init__(self, w_crop, h_crop):
        self.w_crop = w_crop
        self.h_crop = h_crop

    def __call__(self, sample):
        image_orig = sample['raw']
        image_mask = sample['mask']

        crop_orig = image_orig[self.h_crop:(image_orig.shape[0] - self.h_crop),
                    self.w_crop:(image_orig.shape[1] - self.w_crop)]
        crop_mask = image_mask[self.h_crop:(image_mask.shape[0] - self.h_crop),
                    self.w_crop:(image_mask.shape[1] - self.w_crop)]

        return {'raw': crop_orig,
                'mask': crop_mask,
                }


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):

        image_orig = sample['raw']
        image_mask = sample['mask']

        h, w = image_orig.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image_orig, (new_h, new_w))
        mask = transform.resize(image_mask, (new_h, new_w))

        # Enhance image after resize with a binary threshold. Cast to long type since cross entropy loss
        # needs long type values
        mask = np.where(mask > 0.08, 1., 0.).astype(long)

        return {'raw': img,
                'mask': mask,
                }


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image_orig = sample['raw']
        image_mask = sample['mask']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image_orig = image_orig.transpose((2, 0, 1)).astype(float32)
        image_mask = image_mask

        return {'raw': torch.from_numpy(image_orig),
                'mask': torch.from_numpy(image_mask),
                }
