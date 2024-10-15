
from typing import (Generic, TypeVar, Callable, Sequence)

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

Tin = TypeVar('Tin')
Tout = TypeVar('Tout')
    
    
class ImageSpectrumDatasetRS(Dataset):
    def __init__(self, dataset, transform_image=None, transform_spectrum=None, wavelengths=None, fromarray=True):
        self.dataset = dataset
        
        self.transform_image = transform_image
        self.transform_spectrum = transform_spectrum
        self.fromarray = fromarray
        self.wavelengths = wavelengths

    def __getitem__(self, index):
        x, y, ind = self.dataset[index]
        
        if self.transform_image:
            if self.fromarray:
                x = Image.fromarray(x.astype(np.uint8))
            x = self.transform_image(x)
        
        if self.transform_spectrum:
            y = self.transform_spectrum(y, ind[1], self.wavelengths)

        return x, y, ind

    def __len__(self):
        return len(self.dataset)


class FileListDataset(Dataset, Generic[Tin, Tout]):

    def __init__(self,
                 file_list: Sequence[Tin],
                 loader: Callable[[Tin], Tout] = np.load
                 ) -> None:
        self.loader = loader
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tout:
        return self.loader(self.file_list[idx])


def triple_load(x):
    return np.load(x[0]), np.load(x[1]), np.load(x[2])


def normalize(image, train_means, train_stds):
    image = image - train_means[:, None, None]
    image = image / train_stds[:, None, None]
    
    return image