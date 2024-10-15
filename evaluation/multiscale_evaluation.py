
import logging
import importlib
import os
import shutil 

import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import ignite.distributed as idist
from ignite.utils import setup_logger

from utils.utils import expanduservars
from contrastive.trainer import load as con_load, Trainer as con_Trainer
from generative.trainer import load as gen_load, Trainer as gen_Trainer, PolyakAverager, _build_model, _loader_subset
from contrastive.builder import build_model as build_contrastive_model

LOGGER = logging.getLogger(__name__)


def get_diffs(predictions, spectra):
    mse = torch.mean((predictions[:, 0] - spectra[:, 0]) ** 2, dim=1)
    
    return mse


@torch.no_grad()
def evaluate_mean(loader, model, con_model, output_path, params):
    dataset = params["dataset_file"]
    num_lr_to_save = params["num_lr_to_save"]
    num_sr_to_save = params["num_sr_to_save"]
    num_predictions = params["num_eval_predictions"]
    split = 'lr' if 'lr' in dataset else 'sr'

    model.eval()
    con_mses = []

    inds = []
    num_processed = 0
    for batch_idx, batch in enumerate(loader):
        LOGGER.info("Batch %d (%d pairs total)", batch_idx, num_processed + batch[0].shape[0])

        images_b, spectra_b, inds_b = batch
        images_b = images_b.to(idist.device())
        spectra_b = spectra_b.to(idist.device())
        inds.append(inds_b)

        if 'sr' in dataset:
            generated_spectra_b = []
            for sample in range(images_b.shape[0]):
                for spectrum in range(num_lr_to_save):
                    generated_spectra_b.append(np.load(os.path.join(output_path, 'lr', str(num_processed + sample), str(spectrum) + '.npy')))
                
            generated_spectra_b = np.concatenate(generated_spectra_b, axis=0)
            generated_spectra_b = rearrange(generated_spectra_b, '(b s) l -> b s l', b = images_b.shape[0])
            generated_spectra_b = torch.as_tensor(generated_spectra_b).to(idist.device())

            if params["preprocessing"] == 'log':
                generated_spectra_b = torch.log(generated_spectra_b)
            
            num_each = num_predictions / generated_spectra_b.shape[1]

        spectra_b = spectra_b[:, None]
        
        predictions_b = torch.stack(
            [model(None, images_b, None if 'lr' in dataset else generated_spectra_b[:, int(i // num_each)][:, None]) for i in range(num_predictions)],
            dim=1
        )

        img_features, spec_features = get_features(images_b, predictions_b, con_model)
        scores = np.sum(img_features * spec_features, axis=-1)
        sort_indices = np.argsort(scores, axis=-1)
        I, _ = np.ix_(np.arange(images_b.shape[0]), np.arange(num_predictions))

        predictions_con = predictions_b[I, sort_indices]
        predictions_con = predictions_con[:, -num_lr_to_save:] if 'lr' in dataset else predictions_con[:, -num_sr_to_save:]

        if params["preprocessing"] == 'log':
            predictions_con = torch.exp(predictions_con)
            spectra_b = torch.exp(spectra_b)        
        
        for sample in range(images_b.shape[0]):
            directory = os.path.join(output_path, split, str(num_processed + sample))

            if os.path.exists(directory):
                shutil.rmtree(directory)

            os.makedirs(directory, exist_ok=True)
            for spectrum in range(predictions_con.shape[1]):
                np.save(os.path.join(directory, str(spectrum) + '.npy'), predictions_con[sample, spectrum].cpu().numpy())

        predictions_con = predictions_con[:, -1]
        con_mses.append(get_diffs(predictions_con, spectra_b))

        num_processed += images_b.shape[0]

    inds = torch.cat(inds)
    con_mses = torch.cat(con_mses, dim=0)

    return con_mses, inds.numpy()


def get_features(images, predictions, model):
    model.eval()
    img_features = []
    spec_features = []

    with torch.no_grad():
        for i in range(len(images)):
            img, spec = images.to(idist.device()), predictions.to(idist.device())
            out_1, out_2 = model(img[i][None], spec[i, :, 0])

            img_features.append(out_1[None])
            spec_features.append(out_2[None])

    img_features = torch.cat(img_features)
    spec_features = torch.cat(spec_features)
    
    return img_features.cpu().numpy(), spec_features.cpu().numpy()


def _build_datasets(params: dict, output_path):
    dataset_file: str = params['dataset_file']
    dataset_module = importlib.import_module(dataset_file)
    test_dataset = dataset_module.test_dataset(params, output_path)  # type: ignore

    LOGGER.info("%d datapoints in test dataset '%s'", len(test_dataset), dataset_file)

    test_loader = DataLoader(
        test_dataset,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=params["mp_loaders"],
    )

    return test_loader


def ms_eval(local_rank: int, params: dict):

    for dataset in ['datasets.lr_5band', 'datasets.sr_5band']:
        params["dataset_file"] = dataset
        setup_logger(name=None, format="\x1b[32;1m%(asctime)s [%(name)s]\x1b[0m %(message)s", reset=True)

        # Create output folder and archive the current code and the parameters there
        output_path = expanduservars(params['output_path'])
        
        LOGGER.info("%d GPUs available", torch.cuda.device_count())
        
        # Load the datasets
        test_loader = _build_datasets(params, output_path)
        
        # Build the model, optimizer, trainer and training engine
        input_shapes = [i.shape for i in test_loader.dataset[0]]
        LOGGER.info("Input shapes: " + str(input_shapes))

        gen_model, gen_average_model = [_build_model(params, input_shapes) for _ in range(2)]
        con_model = build_contrastive_model(
            params=params
        ).to(idist.device())

        optimizer = torch.optim.Adam(con_model.parameters(), lr=params["learning_rate"])
        scheduler = CosineAnnealingLR(optimizer, T_max = 300)

        con_trainer = con_Trainer(con_model, optimizer, scheduler)

        polyak = PolyakAverager(gen_model, gen_average_model, alpha=params["polyak_alpha"])
        optimizer = torch.optim.Adam(gen_model.parameters(), lr=params["learning_rate"])
        gen_trainer = gen_Trainer(polyak, optimizer, scheduler, params["num_eval_predictions"], params["preprocessing"], params["dataset_file"])

        load_from = params.get('load_lr_con_from' if 'lr' in dataset else 'load_sr_con_from', None)
        if load_from is not None:
            load_from = expanduservars(load_from)
            con_load(load_from, trainer=con_trainer, engine=None)

        load_from = params.get('load_lr_gen_from' if 'lr' in dataset else 'load_sr_gen_from', None)
        if load_from is not None:
            load_from = expanduservars(load_from)
            gen_load(load_from, trainer=gen_trainer, engine=None)

        con_mses, inds = evaluate_mean(_loader_subset(test_loader, params["num_eval_samples"]), gen_average_model, con_model, output_path, params)

        LOGGER.info("MSE: %.3f with %d predictions.", torch.mean(con_mses), params["num_eval_predictions"])