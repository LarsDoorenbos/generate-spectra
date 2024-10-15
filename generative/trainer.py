
from dataclasses import dataclass
import logging
import os
import platform
import importlib
from typing import Iterable, Optional, Tuple, Dict, Union, Any, cast

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from torch import nn
import torch
from torch.utils.data import DataLoader, Subset

from ignite.engine import Engine, Events
import ignite.distributed as idist
from ignite.utils import setup_logger
from ignite.metrics import Frequency, MeanSquaredError
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.contrib.handlers import WandBLogger
from ignite.contrib.metrics import GpuInfo

from .ddpm_builder import DenoisingModel, build_model
from .polyak import PolyakAverager
from .lr_functions import LRFcts
from utils.utils import archive_code, expanduservars, Denormalize

LOGGER = logging.getLogger(__name__)
Model = Union[DenoisingModel, nn.parallel.DataParallel, nn.parallel.DistributedDataParallel]


def _flatten(m: Model) -> DenoisingModel:
    if isinstance(m, DenoisingModel):
        return m
    elif isinstance(m, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)):
        return cast(DenoisingModel, m.module)
    else:
        raise TypeError("type(m) should be one of (DenoisingModel, DataParallel, DistributedDataParallel)")


def _loader_subset(loader: DataLoader, num_images: int) -> DataLoader:
    dataset = loader.dataset
    lng = len(dataset)
    return DataLoader(
        Subset(dataset, range(0, lng - lng % num_images, lng // num_images)),
        batch_size=loader.batch_size,
        shuffle=False
    )


def _get_ylimits(spectra, predictions):
    min_y = torch.min(torch.min(spectra), torch.min(predictions))
    max_y = torch.max(torch.max(spectra), torch.max(predictions))

    return max_y.cpu().numpy(), min_y.cpu().numpy()


@torch.no_grad()
def grid_of_spectrum_predictions(model, loader, num_predictions, dataset, preprocessing):
    model.eval()

    images_= []
    spectra_ = []
    predictions_ = []
    for batch in loader:
        images_b, spectra_b, indices_b = batch
        images_b = images_b.to(idist.device())
        spectra_b = spectra_b.to(idist.device())

        spectra_b = spectra_b[:, None]

        predictions_b = torch.stack(
            [model(None, images_b, spectra_b) if 'sr' in dataset else model(None, images_b) for _ in range(num_predictions)],
            dim=1
        )

        images_.append(images_b)
        spectra_.append(spectra_b)
        predictions_.append(predictions_b)

    images = torch.cat(images_, dim=0)
    spectra = torch.cat(spectra_, dim=0)
    predictions = torch.cat(predictions_, dim=0)

    denorm = Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if images.shape[1] == 3 else Denormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    images = images if images.shape[1] == 3 else images[:, 1:-1]

    xs = np.arange(4000, 8500, 8 if spectra.shape[-1] == 563 else 1)
    _, axarr = plt.subplots(images.shape[0], predictions.shape[1] + 2, figsize=(25, 25))
    for img in range(len(images)):

        axarr[img, 0].imshow(np.moveaxis(denorm(images[img]).cpu().numpy(),0,2))
        axarr[img, 0].axis('off')
        axarr[img, 1].plot(xs, spectra[img, 0].cpu().numpy() if preprocessing != 'log' else np.exp(spectra[img, 0].cpu().numpy()))
        axarr[img, 1].set_ylim(0, 3)
        
        for i in range(predictions.shape[1]):
            axarr[img, 2+i].plot(xs, predictions[img, i, 0, :].cpu().numpy() if preprocessing != 'log' else np.exp(predictions[img, i, 0, :].cpu().numpy()))
            axarr[img, 2+i].set_ylim(0, 3)
                
    return plt


@dataclass
class Trainer:

    polyak: PolyakAverager
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler
    num_eval_predictions: int
    preprocessing: str
    dataset: str

    @property
    def model(self):
        return self.polyak.model

    @property
    def flat_model(self):
        """View of the model without DataParallel wrappers."""
        return _flatten(self.model)

    @property
    def average_model(self):
        return self.polyak.average_model
        
    @property
    def time_steps(self):
        return self.flat_model.time_steps

    @property
    def diffusion_model(self):
        return self.flat_model.diffusion

    def train_step(self, e: Engine, batch) -> dict:
        condition, sample, _ = batch
        sample = sample[:, None]

        self.model.train()

        device = idist.device()
        sample = sample.to(device, non_blocking=True)
        condition = condition.to(device, non_blocking=True)

        batch_size = sample.shape[0]

        # Sample a random step and generate gaussian noise
        t = torch.randint(0, self.time_steps, size=(batch_size,), device=device)
        noise = torch.randn_like(sample)

        t = torch.as_tensor([0], device=device)
        cumalphas_t = self.diffusion_model.cumalphas[t][:, None, None]
        xt = torch.sqrt(cumalphas_t)*sample + torch.sqrt(1 - cumalphas_t)*noise

        # Compute the corresponding x_t for the chosen step and gaussian noise
        cumalphas_t = self.diffusion_model.cumalphas[t][:, None, None]
            
        xt = torch.sqrt(cumalphas_t)*sample + torch.sqrt(1 - cumalphas_t)*noise

        # Estimate the noise with the model
        if 'sr' in self.dataset:
            noise_pred = self.model(xt, condition, sample, t)
        else:
            noise_pred = self.model(xt, condition, t=t)

        # Penalize the difference between real and estimated noise
        loss = nn.functional.mse_loss(noise_pred, noise)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.polyak.update()

        lr = self.scheduler.get_last_lr()[0]
        self.scheduler.step()

        return {"num_items": batch_size, "loss": loss.item(), "lr": lr}

    @torch.no_grad()
    def test_step(self, _: Engine, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        image, spectra, _ = batch
        spectra = spectra[:, None]

        self.average_model.eval()

        device = idist.device()
        spectra = spectra.to(device, non_blocking=True)
        image = image.to(device, non_blocking=True)

        predictions = torch.stack(
            [self.average_model(None, image, spectra) if 'sr' in self.dataset else self.average_model(None, image) for _ in range(self.num_eval_predictions)],
            dim=1
        ) 

        if self.preprocessing == 'log':
            predictions = torch.exp(predictions)
            spectra = torch.exp(spectra)

        # Compute MSEs between predictions and ground truth
        mse = nn.functional.mse_loss(predictions, spectra[:, None], reduction="none")[:, :, 0]
        mse = torch.sum(mse, dim=2)
        mse = torch.argmin(mse, dim=1)
        
        # Select the best prediction for each spectrum
        predictions = predictions[torch.arange(predictions.shape[0]), mse]
        
        return {"y": spectra, "y_pred": predictions}

    def objects_to_save(self, engine: Optional[Engine] = None) -> dict:
        to_save: Dict[str, Any] = {
            "model": self.flat_model,
            "average_model": _flatten(self.average_model),
            "optimizer": self.optimizer,
            "scheduler": self.scheduler
        }

        if engine is not None:
            to_save["engine"] = engine

        return to_save


def build_engine(trainer: Trainer, output_path: str, validation_loader: Iterable, params: dict) -> Engine:

    engine = Engine(trainer.train_step)
    frequency_metric = Frequency(output_transform=lambda x: x["num_items"])
    frequency_metric.attach(engine, "imgs/s", Events.ITERATION_COMPLETED)
    GpuInfo().attach(engine, "gpu")

    engine_test = Engine(trainer.test_step)
    MeanSquaredError().attach(engine_test, 'mse')

    if idist.get_local_rank() == 0:
        
        if params["use_logger"]:
            wandb_logger = WandBLogger(project='generate-spectra', config=params)
        
            wandb_logger.attach_output_handler(
                engine,
                Events.ITERATION_COMPLETED(every=100),
                tag="training",
                output_transform=lambda x: x,
                metric_names=["imgs/s"],
                global_step_transform=global_step_from_engine(engine, Events.ITERATION_COMPLETED)
            )

            wandb_logger.attach_output_handler(
                engine_test,
                Events.EPOCH_COMPLETED,
                tag="testing",
                metric_names=["mse"],
                global_step_transform=global_step_from_engine(engine, Events.ITERATION_COMPLETED)
            )
        else: 
            wandb_logger = None

        checkpoint_handler = ModelCheckpoint(
            output_path,
            "model",
            n_saved=2,
            require_empty=False,
            score_function=None,
            score_name=None
        )

        checkpoint_best = ModelCheckpoint(
            output_path,
            "best",
            n_saved=2,
            require_empty=False,
            score_function=lambda engine: float("inf") if np.isnan(engine_test.state.metrics["mse"]) or engine_test.state.metrics["mse"] == float("-inf") else -engine.state.metrics['mse'],
            score_name='negmse',
            global_step_transform=global_step_from_engine(engine, Events.ITERATION_COMPLETED)
        )
    

    # Display some info every 100 iterations
    @engine.on(Events.ITERATION_COMPLETED(every=100))
    @idist.one_rank_only(rank=0, with_barrier=True)
    def log_info(engine: Engine):
        LOGGER.info(
            "epoch=%d, iter=%d, speed=%.2fimg/s, loss=%.4g, lr=%g, gpu:0 util=%.2f%%",
            engine.state.epoch,
            engine.state.iteration,
            engine.state.metrics["imgs/s"],
            engine.state.output["loss"],
            engine.state.output["lr"],
            engine.state.metrics["gpu:0 util(%)"]
        )

    # Save model every 1000 iterations
    @engine.on(Events.ITERATION_COMPLETED(every=1000))
    @idist.one_rank_only(rank=0, with_barrier=True)
    def save_model(engine: Engine):
        checkpoint_handler(engine, trainer.objects_to_save(engine))

    # Generate and save a few segmentations every 5000 iterations
    @engine.on(Events.ITERATION_COMPLETED(every=5000))
    @idist.one_rank_only(rank=0, with_barrier=True)
    def save_qualitative_results(_: Engine, num_images=5, num_predictions=3):
        LOGGER.info("Generating samples...")
        loader = _loader_subset(validation_loader, num_images)

        plt = grid_of_spectrum_predictions(_flatten(trainer.average_model), loader, num_predictions, params["dataset_file"], params["preprocessing"])

        filename = os.path.join(output_path, f"samples_{engine.state.iteration:06}.png")
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.0)
        plt.close('all')

    # Compute the MSE score every 5000 iterations
    @engine.on(Events.ITERATION_COMPLETED(every=5000))
    def compute_mse(_: Engine):
        LOGGER.info("MSE computation...")
        engine_test.run(_loader_subset(validation_loader, 256), max_epochs=1)
        
    # Save the best models by MSE score
    @engine_test.on(Events.EPOCH_COMPLETED)
    @idist.one_rank_only(rank=0, with_barrier=True)
    def log_mse(engine_test: Engine):
        LOGGER.info("MSE score: %.4g", engine_test.state.metrics["mse"])
        checkpoint_best(engine_test, trainer.objects_to_save(engine))

    return engine


def load(filename: str, trainer: Trainer, engine: Engine):
    LOGGER.info("Loading state from %s...", filename)
    state = torch.load(filename, map_location=idist.device())
    to_load = trainer.objects_to_save(engine)
    ModelCheckpoint.load_objects(to_load, state)


def _build_model(params: dict, item_shapes: Tuple[int, int, int]) -> Model:
    model: Model = build_model(
        time_steps=params["time_steps"],
        schedule=params["beta_schedule"],
        conditioning=params["conditioning"],
        dataset=params["dataset_file"],
        reduction=params["reduction"],
        backbone_params=params[params["backbone"]],
        item_shapes=item_shapes
    ).to(idist.device())

    # Wrap the model in DataParallel for parallel processing
    if params["distributed"]:
        local_rank = idist.get_local_rank()
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    elif params["multigpu"]:
        model = nn.DataParallel(model)

    return model


def _build_datasets(params: dict, output_path) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    dataset_file: str = params['dataset_file']
    dataset_module = importlib.import_module(dataset_file)
    train_dataset, validation_dataset = dataset_module.training_dataset(params, output_path)  # type: ignore

    LOGGER.info("%d datapoints in dataset '%s'", len(train_dataset), dataset_file)
    LOGGER.info("%d datapoints in validation dataset '%s'", len(validation_dataset), dataset_file)

    dataset_loader = DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=params["mp_loaders"],
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=params["mp_loaders"],
    )

    return dataset_loader, validation_loader


def train_ddpm(local_rank: int, params: dict):

    setup_logger(name=None, format="\x1b[32;1m%(asctime)s [%(name)s]\x1b[0m %(message)s", reset=True)
    LOGGER.setLevel(logging.INFO)

    # Create output folder and archive the current code and the parameters there
    output_path = expanduservars(params['output_path'])
    os.makedirs(output_path, exist_ok=True)
    archive_code(output_path, 'generative')

    num_gpus = torch.cuda.device_count()
    LOGGER.info("%d GPUs available", num_gpus)

    # Load the datasets
    train_loader, validation_loader = _build_datasets(params, output_path)

    # Build the model, optimizer, trainer and training engine
    input_shapes = [train_loader.dataset[5][0].shape, train_loader.dataset[5][1].shape]
    LOGGER.info("Input shapes: " + str(input_shapes))

    model, average_model = [_build_model(params, input_shapes).float() for _ in range(2)]
    average_model.load_state_dict(model.state_dict())
    
    polyak = PolyakAverager(model, average_model, alpha=params["polyak_alpha"])

    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

    num_batches_per_epoch = len(train_loader)  # fixme what happens in multigpu (wabout device-wise batch size?)
    lr_total_steps = num_batches_per_epoch * params["max_epochs"]

    lr_restart_steps = []  # todo add lr_restarts code for now defaulting to not using restarts

    p_opt = params['optim']
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LRFcts(p_opt, lr_total_steps,
                                                                                        lr_restart_steps))

    LOGGER.info("lr_schedule: '{}' starting lr {} and lr_params {} over total steps {}".format(p_opt['lr_function'], p_opt['learning_rate'], p_opt['lr_params'], lr_total_steps))
        
    trainer = Trainer(polyak, optimizer, scheduler, params["num_eval_predictions"], params["preprocessing"], params["dataset_file"])
    engine = build_engine(trainer, output_path, validation_loader, params)

    # Load a model (if requested in params.yml) to continue training from it
    load_from = params.get('load_from', None)
    if load_from is not None:
        load_from = expanduservars(load_from)
        load(load_from, trainer=trainer, engine=engine)
        optimizer.param_groups[0]['capturable'] = True

    # Run the training engine for the requested number of epochs
    engine.run(train_loader, max_epochs=params["max_epochs"])