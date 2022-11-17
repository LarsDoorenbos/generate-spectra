
import os
import logging

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

import ignite.distributed as idist
from ignite.utils import setup_logger

from utils.utils import expanduservars
from contrastive.trainer import load as con_load, Trainer as con_Trainer
from generative.trainer import load as gen_load, Trainer as gen_Trainer, PolyakAverager, _get_ylimits, _build_datasets, _build_model
from contrastive.builder import build_model as build_contrastive_model

LOGGER = logging.getLogger(__name__)


def _loader_subset(loader: DataLoader, num_images: int) -> DataLoader:
    dataset = loader.dataset
    lng = len(dataset)
    return DataLoader(
        Subset(dataset, range(0, lng - lng % num_images, lng // num_images)),
        batch_size=loader.batch_size,
        shuffle=False
    )


@torch.no_grad()
def get_predictions(model, loader, num_predictions):
    model.eval()
    images_= []
    spectra_ = []
    predictions_ = []
    for batch in loader:
        images_b, spectra_b, _ = batch
        images_b = images_b.to(idist.device())
        spectra_b = spectra_b.to(idist.device())

        spectra_b = spectra_b[:, None]

        predictions_b = torch.stack(
            [model(None, images_b) for _ in range(num_predictions)],
            dim=1
        )

        images_.append(images_b)
        spectra_.append(spectra_b)
        predictions_.append(predictions_b)

    images = torch.cat(images_, dim=0)
    spectra = torch.cat(spectra_, dim=0)
    predictions = torch.cat(predictions_, dim=0)

    return images, spectra, predictions


def get_diffs(predictions, spectra):
    mses = []

    mse = torch.mean((predictions[:, 0] - spectra[:, 0]) ** 2, dim=1)
    mses.append(mse)
    mses = torch.cat(mses)
    
    return mses


@torch.no_grad()
def evaluate_mean(loader, model, num_predictions, con_model, num_con_mean):
    model.eval()

    for batch in loader:
        images_b, spectra_b, _ = batch
        images_b = images_b.to(idist.device())
        spectra_b = spectra_b.to(idist.device())

        spectra_b = spectra_b[:, None]

        predictions_b = torch.stack(
            [model(None, images_b) for _ in range(num_predictions)],
            dim=1
        )
        predictions_mean = torch.mean(predictions_b, dim=1)
        mses = get_diffs(predictions_mean, spectra_b)

        img_features, spec_features = get_features(images_b, predictions_b, con_model)
        scores = np.sum(img_features * spec_features, axis=-1)
        sort_indices = [np.arange(predictions_b.shape[0]), np.argmax(scores, axis=-1)]

        predictions_con = predictions_b[sort_indices]
        con_mses = get_diffs(predictions_con, spectra_b)

        sort_indices = np.argsort(scores, axis=-1)[:, -num_con_mean:]
        predictions_con_mean = torch.mean(np.take_along_axis(predictions_b, sort_indices[:, :, None, None], axis=1), dim=1)
        con_mean_mses = get_diffs(predictions_con_mean, spectra_b)

    return mses, con_mses, con_mean_mses


def plot_samples(loader, gen_trainer, num_eval_predictions, num_eval_samples):
    loader = _loader_subset(loader, num_eval_samples)
    images, spectra, predictions = get_predictions(gen_trainer.average_model, loader, num_eval_predictions)

    return images, spectra, predictions


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


def rank_samples(images, spectra, predictions, con_trainer, output_path, num_shown, num_con_mean):
    loglam = np.load('data/galaxies/loglam_uniform1.npz', allow_pickle=True)['arr_0'][0]
    
    img_features, spec_features = get_features(images, predictions, con_trainer.model)
    images = images.cpu() * torch.tensor([0.229, 0.224, 0.225])[:, None, None] + torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    for img in range(len(images)):
        scores = np.sum(img_features[img] * spec_features[img], axis=-1)
        sort_indices = np.argsort(-1*scores)

        f, axarr = plt.subplots(1, num_shown + 4, figsize=(35, 8))

        max_y, min_y = _get_ylimits(spectra[img], predictions[img][sort_indices[:num_shown]])

        axarr[0].imshow(np.moveaxis((images)[img].numpy(),0,2))
        axarr[0].axis('off')
        axarr[1].plot(spectra[img, 0].cpu().numpy())
        axarr[1].set_ylim(min_y, max_y)

        axarr[2].plot(loglam, np.mean(predictions[img].cpu().numpy(), axis=0)[0])
        axarr[2].set_ylim(min_y, max_y)
        mse = np.mean((np.mean(predictions[img].cpu().numpy(), axis=0)[0] - spectra[img, 0].cpu().numpy()) ** 2)
        axarr[2].set_title(str("{:.2f}".format(mse)))

        axarr[3].plot(loglam, np.mean(predictions[img, sort_indices[:num_con_mean], 0, :].cpu().numpy(), axis=0))
        axarr[3].set_ylim(min_y, max_y)
        mse = np.mean((np.mean(predictions[img, sort_indices[:num_con_mean], 0, :].cpu().numpy(), axis=0) - spectra[img, 0].cpu().numpy()) ** 2)
        axarr[3].set_title(str("{:.2f}".format(mse)))
        
        for i in range(num_shown):
            axarr[4+i].plot(loglam, predictions[img, sort_indices[i], 0, :].cpu().numpy())
            axarr[4+i].set_ylim(min_y, max_y)
            mse = np.mean((predictions[img, sort_indices[i], 0, :].cpu().numpy() - spectra[img, 0].cpu().numpy()) ** 2)
            axarr[4+i].set_title(str("{:.2f} {:.2f}".format(scores[sort_indices[i]], mse)))
                
        plt.subplots_adjust(wspace=0, hspace=0)
        filename = os.path.join(output_path, f"ranked_samples" + str(img) + ".png")
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.0)
        plt.close('all')


def generate_and_rank(local_rank: int, params: dict):
    setup_logger(name=None, format="\x1b[32;1m%(asctime)s [%(name)s]\x1b[0m %(message)s", reset=True)

    # Create output folder and archive the current code and the parameters there
    output_path = expanduservars(params['output_path'])
    
    LOGGER.info("%d GPUs available", torch.cuda.device_count())

    # Load the datasets
    _, validation_loader = _build_datasets(params)

    # Build the model, optimizer, trainer and training engine
    input_shapes = [i.shape for i in validation_loader.dataset[0]]
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
    gen_trainer = gen_Trainer(polyak, optimizer, scheduler, params["num_eval_predictions"])

    load_from = params.get('load_con_from', None)
    if load_from is not None:
        load_from = expanduservars(load_from)
        con_load(load_from, trainer=con_trainer, engine=None)

    load_from = params.get('load_from', None)
    if load_from is not None:
        load_from = expanduservars(load_from)
        gen_load(load_from, trainer=gen_trainer, engine=None)

    mses, con_mses, con_mean_mses = evaluate_mean(validation_loader, gen_average_model, params["num_eval_predictions"], con_model, params["num_con_mean"])
    LOGGER.info("Mean MSE: %.3f with %d predictions", torch.mean(mses), params["num_eval_predictions"])
    LOGGER.info("Contrastive MSE: %.3f with %d predictions", torch.mean(con_mses), params["num_eval_predictions"])
    LOGGER.info("Contrastive mean MSE: %.3f with %d predictions", torch.mean(con_mean_mses), params["num_eval_predictions"])

    images, spectra, predictions = plot_samples(validation_loader, gen_trainer, params["num_eval_predictions"], params["num_eval_samples"])
    rank_samples(images, spectra, predictions, con_trainer, output_path, params["num_shown"], params["num_con_mean"])
    
