
import glob

import numpy as np

import torch
import torchvision.transforms as transforms

from .utils import ImageSpectrumDataset, FileListDataset

BASE_PATH = "data/"

transform_image = transforms.Compose([
        transforms.Resize(71),
        transforms.RandomCrop(64),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

transform_image_eval = transforms.Compose([
        transforms.Resize(71),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]) 


def scale_spectrum(spectrum):
    spectrum = (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum))
    spectrum = (spectrum - 0.5) * 2
    return spectrum


def training_dataset(preprocessing):
    img_file_list = sorted(glob.glob(BASE_PATH + "galaxies/data_files/train/images/*.npy"))
    spec_file_list = sorted(glob.glob(BASE_PATH + "galaxies/data_files/train/spectra/*.npy"))
    ind_file_list = sorted(glob.glob(BASE_PATH + "galaxies/data_files/train/indices/*.npy"))

    train_dataset = FileListDataset(list(zip(img_file_list, spec_file_list, ind_file_list)), loader=lambda x: (np.load(x[0]), np.load(x[1]), np.load(x[2])))

    img_file_list = sorted(glob.glob(BASE_PATH + "galaxies/data_files/val/images/*.npy"))
    spec_file_list = sorted(glob.glob(BASE_PATH + "galaxies/data_files/val/spectra/*.npy"))
    ind_file_list = sorted(glob.glob(BASE_PATH + "galaxies/data_files/val/indices/*.npy"))

    val_dataset = FileListDataset(list(zip(img_file_list, spec_file_list, ind_file_list)), loader=lambda x: (np.load(x[0]), np.load(x[1]), np.load(x[2])))

    if preprocessing == 'scale':
        transform_spectrum = scale_spectrum
    else:
        transform_spectrum = lambda x: x

    train_dataset = ImageSpectrumDataset(train_dataset, transform_image, transform_spectrum)
    val_dataset = ImageSpectrumDataset(val_dataset, transform_image_eval, transform_spectrum)

    val_dataset, _ = torch.utils.data.random_split(val_dataset, [5000, len(val_dataset)-5000], generator=torch.Generator().manual_seed(1))

    return train_dataset, val_dataset