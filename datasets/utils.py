
from typing import (Generic, TypeVar, Callable, Sequence,
                    Pattern, Union, List, Dict, Any, cast)

import numpy as np
from PIL import Image

from torch.utils.data import Dataset

Tin = TypeVar('Tin')
Tout = TypeVar('Tout')


class ImageSpectrumDataset(Dataset):
    def __init__(self, dataset, transform_image=None, transform_spectrum=None):
        self.dataset = dataset
        
        self.transform_image = transform_image
        self.transform_spectrum = transform_spectrum

    def __getitem__(self, index):
        x, y, ind = self.dataset[index]
        
        if self.transform_image:
            x = Image.fromarray(x.astype(np.uint8))
            x = self.transform_image(x)
        
        if self.transform_spectrum:
            y = self.transform_spectrum(y)

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