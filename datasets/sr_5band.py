
import glob
from functools import partial

import numpy as np
from scipy.interpolate import CubicSpline
from astropy.convolution import Gaussian1DKernel, convolve

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf

from .utils import ImageSpectrumDatasetRS, FileListDataset, triple_load, normalize


BASE_PATH = "../spectra-generation/data/5band/data_files/"
LOGLAM_BASE_PATH = "../spectra-generation/data/5band/"

g = Gaussian1DKernel(stddev=2)


def transform_image(img_res, train_means, train_stds, image):
    image = torch.as_tensor(image)
    image = image.permute(2, 0, 1)

    if image.shape[1] != 64:
        image = tf.resize(image, img_res, interpolation=tf.InterpolationMode.BILINEAR, antialias=True)
        image = tf.resize(image, 64, interpolation=tf.InterpolationMode.BILINEAR, antialias=True)

    image = tf.resize(image, 71, interpolation=tf.InterpolationMode.BILINEAR, antialias=True)

    i, j, th, tw = transforms.RandomCrop.get_params(image, (64, 64))
    image = tf.crop(image, i, j, th, tw)

    if torch.rand(1) > 0.5:
        image = tf.hflip(image)

    if torch.rand(1) > 0.5:
        image = tf.vflip(image)

    image = normalize(image, train_means, train_stds)
    
    return image.float()


def transform_image_eval(img_res, train_means, train_stds, image):
    image = torch.as_tensor(image)
    image = image.permute(2, 0, 1)

    if image.shape[1] != 64:
        image = tf.resize(image, img_res, interpolation=tf.InterpolationMode.BILINEAR, antialias=True)
        image = tf.resize(image, 64, interpolation=tf.InterpolationMode.BILINEAR, antialias=True)

    image = tf.resize(image, 71, interpolation=tf.InterpolationMode.BILINEAR, antialias=True)

    image = tf.center_crop(image, (64, 64))

    image = normalize(image, train_means, train_stds)

    return image


def process_spectrum(preprocessing, spectrum, redshift, wavelengths):
    corrected_spectrum = spectrum

    corrected_spectrum = corrected_spectrum[wavelengths < 8500]
    wavelengths = wavelengths[wavelengths < 8500]
    
    corrected_spectrum = corrected_spectrum[wavelengths > 4000]
    wavelengths = wavelengths[wavelengths > 4000]

    cs = CubicSpline(wavelengths, corrected_spectrum)
    interpolated_spectrum = cs(np.arange(np.round(np.min(wavelengths)), np.round(np.max(wavelengths))))

    smoothed_spectrum = convolve(interpolated_spectrum, g)

    norm_factor = np.mean(smoothed_spectrum[int(6900 - np.round(np.min(wavelengths))):int(6950 - np.round(np.min(wavelengths)))])
    normalized_spectrum = smoothed_spectrum / norm_factor

    if preprocessing == 'log':
        normalized_spectrum = np.log(normalized_spectrum)
        
    return torch.as_tensor(normalized_spectrum).float()


def training_dataset(params, output_path):
    img_file_list = np.array(sorted(glob.glob(BASE_PATH + "005-015/images/*.npy")))
    spec_file_list = np.array(sorted(glob.glob(BASE_PATH + "005-015/spectra/*.npy")))
    ind_file_list = np.array(sorted(glob.glob(BASE_PATH + "005-015/indices/*.npy")))

    loglam = np.load(LOGLAM_BASE_PATH + 'loglam_uniform1.npz', allow_pickle=True)['arr_0'][0]
    loglam = 10 ** loglam

    train_dataset = FileListDataset(list(zip(img_file_list, spec_file_list, ind_file_list)), loader=triple_load)
    temp_dataset = ImageSpectrumDatasetRS(train_dataset, lambda x: x, partial(process_spectrum, 'none'), loglam, fromarray=False)

    # Select only samples whose spectra have only positive values and get the means and stds of the image channels for normalization
    # TODO: speed up this process
    indices = []
    images = []
    for idx, sample in enumerate(temp_dataset):
        if torch.min(sample[1]) > 0:
            indices.append(idx)
            images.append(sample[0])

    train_means = np.mean(np.array(images), axis=(0, 1, 2))
    train_stds = np.std(np.array(images), axis=(0, 1, 2))

    indices = np.array(indices)
    
    train_dataset = FileListDataset(list(zip(img_file_list[indices], spec_file_list[indices], ind_file_list[indices])), loader=triple_load)

    # 10% + 512 for validation & test
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [len(train_dataset) - int(0.1 * len(train_dataset) + 512), int(0.1 * len(train_dataset) + 512)], generator=torch.Generator().manual_seed(1))
    val_dataset, test_dataset = torch.utils.data.random_split(val_dataset, [512, len(val_dataset) - 512], generator=torch.Generator().manual_seed(1))

    train_dataset = ImageSpectrumDatasetRS(train_dataset, partial(transform_image, params["img_res"], train_means, train_stds), partial(process_spectrum, params["preprocessing"]), loglam, fromarray=False)
    val_dataset = ImageSpectrumDatasetRS(val_dataset, partial(transform_image_eval, params["img_res"], train_means, train_stds), partial(process_spectrum, params["preprocessing"]), loglam, fromarray=False)

    return train_dataset, val_dataset


def test_dataset(params, output_path):
    img_file_list = np.array(sorted(glob.glob(BASE_PATH + "005-015/images/*.npy")))
    spec_file_list = np.array(sorted(glob.glob(BASE_PATH + "005-015/spectra/*.npy")))
    ind_file_list = np.array(sorted(glob.glob(BASE_PATH + "005-015/indices/*.npy")))

    loglam = np.load(LOGLAM_BASE_PATH + 'loglam_uniform1.npz', allow_pickle=True)['arr_0'][0]
    loglam = 10 ** loglam

    train_dataset = FileListDataset(list(zip(img_file_list, spec_file_list, ind_file_list)), loader=triple_load)
    temp_dataset = ImageSpectrumDatasetRS(train_dataset, lambda x: x, partial(process_spectrum, 'none'), loglam, fromarray=False)

    indices = []
    images = []
    for idx, sample in enumerate(temp_dataset):
        if torch.min(sample[1]) > 0:
            indices.append(idx)
            images.append(sample[0])

    train_means = np.mean(np.array(images), axis=(0, 1, 2))
    train_stds = np.std(np.array(images), axis=(0, 1, 2))

    indices = np.array(indices)
    
    train_dataset = FileListDataset(list(zip(img_file_list[indices], spec_file_list[indices], ind_file_list[indices])), loader=triple_load)

    # 10% + 512 for validation & test
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [len(train_dataset) - int(0.1 * len(train_dataset) + 512), int(0.1 * len(train_dataset) + 512)], generator=torch.Generator().manual_seed(1))
    val_dataset, test_dataset = torch.utils.data.random_split(val_dataset, [512, len(val_dataset) - 512], generator=torch.Generator().manual_seed(1))

    test_dataset = ImageSpectrumDatasetRS(test_dataset, partial(transform_image_eval, params["img_res"], train_means, train_stds), partial(process_spectrum, params["preprocessing"]), loglam, fromarray=False)
    return test_dataset