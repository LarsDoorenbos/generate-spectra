
from dataclasses import dataclass
import logging
import importlib
import os
from typing import Optional, Tuple, Dict, Union, Any, cast
import random

import numpy as np
import matplotlib.pyplot as plt

# Torch imports
from torch import nn
from torch import Tensor
import torch
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import MultiplicativeLR
import torch.nn.functional as F

# Ignite imports
from ignite.engine import Engine, Events
import ignite.distributed as idist
from ignite.utils import setup_logger
from ignite.metrics import Frequency
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.contrib.handlers import WandBLogger, ProgressBar
from ignite.contrib.metrics import GpuInfo

# Local imports
from .builder import build_model, MultimodalModel
from utils.utils import archive_code, expanduservars


LOGGER = logging.getLogger(__name__)
Model = Union[MultimodalModel, nn.parallel.DataParallel, nn.parallel.DistributedDataParallel]


def _loader_subset(loader: DataLoader, num_images: int) -> DataLoader:
    dataset = loader.dataset
    lng = len(dataset)
    return DataLoader(
        Subset(dataset, range(0, lng - lng % num_images, lng // num_images)),
        batch_size=loader.batch_size,
        shuffle=False
    )


def _flatten(m: Model) -> MultimodalModel:
    if isinstance(m, MultimodalModel):
        return m
    elif isinstance(m, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)):
        return cast(MultimodalModel, m.module)
    else:
        raise TypeError("type(m) should be one of (MultimodalModel, DataParallel, DistributedDataParallel)")


def get_features(loader, model):
    model.eval()
    
    img_features = []
    spec_features = []
    with torch.no_grad():
        for _, (img, spec, _) in enumerate(loader):
            img, spec = img.to(idist.device()), spec.to(idist.device())

            out_1, out_2 = model(img, spec)

            img_features.append(out_1)
            spec_features.append(out_2)

    img_features = torch.cat(img_features)
    spec_features = torch.cat(spec_features)

    return img_features, spec_features


@torch.no_grad()
def visualize(model, loader, num_images):
    img_features, spec_features = get_features(loader, model)
    
    vis_ind = random.randint(0, len(spec_features) - 1)
    scores = torch.sum(img_features * spec_features[vis_ind], dim=-1)
    
    indices = np.argpartition(scores.cpu(), -num_images)[-num_images:]

    f, axarr = plt.subplots(2, num_images+1, figsize=(27, 9))

    max_y = -10000
    min_y = 10000

    img, spec, _ = loader.dataset[vis_ind]
    spec = spec.numpy()

    if np.max(spec) > max_y:
        max_y = np.max(spec)
    if np.min(spec) < min_y:
        min_y = np.min(spec)

    img = img * torch.tensor([0.229, 0.224, 0.225])[:, None, None] + torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    axarr[0, 0].imshow(np.moveaxis(img.numpy(),0,2))
    axarr[0, 0].axis('off')
    axarr[1, 0].plot(spec)
    
    for i, ind in enumerate(indices):
        img, spec, _ = loader.dataset[ind]        
        spec = spec.numpy()

        img = img * torch.tensor([0.229, 0.224, 0.225])[:, None, None] + torch.tensor([0.485, 0.456, 0.406])[:, None, None]

        axarr[0, i+1].imshow(np.moveaxis(img.numpy(),0,2))
        axarr[0, i+1].axis('off')
        axarr[1, i+1].plot(spec)

        if np.max(spec) > max_y:
            max_y = np.max(spec)
        if np.min(spec) < min_y:
            min_y = np.min(spec)

    for i in range(num_images + 1):
        axarr[1, i].set_ylim(min_y, max_y)
            
    plt.subplots_adjust(wspace=0, hspace=0)

    return plt


def contrastive_loss(out_1, out_2):
    bs = out_1.size(0)
    
    out = torch.cat([out_1, out_2], dim=0)
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / 0.05)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * bs, device=sim_matrix.device)).bool()
    
    sim_matrix = sim_matrix.masked_select(mask).view(2 * bs, -1)

    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / 0.05)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

    return loss


@dataclass
class Trainer:

    model: MultimodalModel
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler

    @property
    def flat_model(self):
        """View of the model without DataParallel wrappers."""
        return _flatten(self.model)

    def train_step(self, engine: Engine, batch) -> dict:

        image, spectrum, _ = batch

        self.model.train()
        
        device = idist.device()
        image = image.to(device, non_blocking=True)
        spectrum = spectrum.to(device, non_blocking=True)

        batch_size = image.shape[0]

        out_1, out_2 = self.model(image, spectrum)
        loss = contrastive_loss(out_1, out_2)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"num_items": batch_size, "loss": loss.item()}

    @torch.no_grad()
    def test_step(self, _: Engine, batch: Tensor) -> Dict[str, Any]:

        image, spectrum, _ = batch
        device = idist.device()
        image = image.to(device, non_blocking=True)
        spectrum = spectrum.to(device, non_blocking=True)

        self.model.eval()

        out_1, out_2 = self.model(image, spectrum)
        loss = contrastive_loss(out_1, out_2)

        top_1 = torch.sum(torch.argmax(torch.sum(out_1[:, None] * out_2[None], dim=-1), dim=-1) == torch.arange(0, out_1.shape[0]).to(device))

        return {"loss": loss.item(), "top1": top_1}

    def objects_to_save(self, engine: Optional[Engine] = None) -> Dict[str, Any]:
        to_save: Dict[str, Any] = {
            "model": self.flat_model,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
        }

        if engine is not None:
            to_save["engine"] = engine

        return to_save


def build_engine(trainer: Trainer, output_path: str, validation_loader: DataLoader, params: dict) -> Engine:
    engine = Engine(trainer.train_step)
    frequency_metric = Frequency(output_transform=lambda x: x["num_items"])
    frequency_metric.attach(engine, "imgs/s", Events.ITERATION_COMPLETED)
    GpuInfo().attach(engine, "gpu")

    engine_test = Engine(trainer.test_step)
    

    if idist.get_local_rank() == 0:
        ProgressBar(persist=True).attach(engine_test)

        if params["use_logger"]:
            wandb_logger = WandBLogger(project='generate-spectra')

            wandb_logger.attach_output_handler(
                engine,
                event_name=Events.ITERATION_COMPLETED(every=100),
                tag="training",
                output_transform=lambda x: x,
                global_step_transform=global_step_from_engine(engine, Events.ITERATION_COMPLETED)
            )

            wandb_logger.attach_output_handler(
                engine_test,
                Events.EPOCH_COMPLETED,
                tag="validation",
                output_transform=lambda x: x,
                global_step_transform=global_step_from_engine(engine, Events.ITERATION_COMPLETED)
            )
        else: 
            wandb_logger = None

        checkpoint_handler = ModelCheckpoint(
            output_path,
            "model",
            n_saved=1,
            require_empty=False,
            score_function=None,
            score_name=None
        )

        checkpoint_best = ModelCheckpoint(
            output_path,
            "best",
            n_saved=2,
            require_empty=False,
            score_function=lambda engine: -engine.state.output["loss"],
            score_name='val_loss',
            global_step_transform=global_step_from_engine(engine, Events.ITERATION_COMPLETED)
        )

    # Display some info every 100 iterations
    @engine.on(Events.ITERATION_COMPLETED(every=100))
    @idist.one_rank_only(rank=0, with_barrier=True)
    def log_info(engine: Engine):
        LOGGER.info(
            "epoch=%d, iter=%d, speed=%.2fimg/s, loss=%.4g, gpu:0 util=%.2f%%",
            engine.state.epoch,
            engine.state.iteration,
            engine.state.metrics["imgs/s"],
            engine.state.output["loss"],
            engine.state.metrics["gpu:0 util(%)"]
        )

    # Visualize every 1000 iterations
    @engine.on(Events.ITERATION_COMPLETED(every=1000))
    @idist.one_rank_only(rank=0, with_barrier=True)
    def save_qualitative_results(_: Engine, num_images=5):
        LOGGER.info("Visualizing...")
        loader = _loader_subset(validation_loader, 2048)
        plt = visualize(_flatten(trainer.model), loader, num_images)
        filename = os.path.join(output_path, f"result_{engine.state.iteration:06}.png")
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.0)
        plt.close('all')

    # Save model every 1000 iterations
    @engine.on(Events.ITERATION_COMPLETED(every=1000))
    @idist.one_rank_only(rank=0, with_barrier=True)
    def save_model(engine: Engine):
        checkpoint_handler(engine, trainer.objects_to_save(engine))

    # Scheduler step
    @engine.on(Events.EPOCH_COMPLETED)
    @idist.one_rank_only(rank=0, with_barrier=True)
    def scheduler_step(_: Engine):
        trainer.scheduler.step()

    # Compute the val loss every 1000 iterations
    @engine.on(Events.ITERATION_COMPLETED(every=1000))
    def test(_: Engine):
        LOGGER.info("Validation loss computation...")
        loader = _loader_subset(validation_loader, params['batch_size'] * 2)
        engine_test.run(loader, max_epochs=1)

    # Save the best models by val loss
    @engine_test.on(Events.EPOCH_COMPLETED)
    @idist.one_rank_only(rank=0, with_barrier=True)
    def log_loss(engine_test: Engine):
        LOGGER.info("Val loss: %.4g", engine_test.state.output["loss"])
        checkpoint_best(engine_test, trainer.objects_to_save(engine))

    return engine


def load(filename: str, trainer: Trainer, engine: Engine):
    LOGGER.info("Loading state from %s...", filename)
    state = torch.load(filename, map_location=idist.device())
    to_load = trainer.objects_to_save(engine)
    ModelCheckpoint.load_objects(to_load, state)


def _build_model(params: dict) -> Model:
    model: Model = build_model(
        params=params
    ).to(idist.device())

    if params["multigpu"]:
        model = nn.DataParallel(model)

    return model


def _build_datasets(params: dict) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    dataset_file: str = params['dataset_file']
    dataset_module = importlib.import_module(dataset_file)
    train_dataset, validation_dataset = dataset_module.training_dataset(params["preprocessing"])  # type: ignore

    LOGGER.info("%d datapoints in dataset '%s'", len(train_dataset), dataset_file)
    LOGGER.info("%d datapoints in validation dataset '%s'", len(validation_dataset), dataset_file)

    dataset_loader = DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=params["mp_loaders"],
        drop_last=True
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=params['batch_size'] * 2,
        shuffle=False,
        num_workers=params["mp_loaders"],
        drop_last=True
    )

    return dataset_loader, validation_loader


def train_contrastive(local_rank: int, params: dict):
    setup_logger(name=None, format="\x1b[32;1m%(asctime)s [%(name)s]\x1b[0m %(message)s", reset=True)

    # Create output folder and archive the current code and the parameters there
    output_path = expanduservars(params['output_path'])
    os.makedirs(output_path, exist_ok=True)
    archive_code(output_path, 'contrastive')

    LOGGER.info("%d GPUs available", torch.cuda.device_count())

    # Load the datasets
    train_loader, validation_loader = _build_datasets(params)

    # Build the model, optimizer, trainer and training engine
    input_shapes = [i.shape for i in train_loader.dataset[0]]
    LOGGER.info("Input shapes: " + str(input_shapes))

    model = _build_model(params)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    scheduler = MultiplicativeLR(optimizer, lr_lambda=lambda x: params["scheduler_lambda"])

    trainer = Trainer(model, optimizer, scheduler)
    engine = build_engine(trainer, output_path, validation_loader, params=params)

    # Load a model (if requested in params.yml) to continue training from it
    load_from = params.get('load_from', None)
    if load_from is not None:
        load_from = expanduservars(load_from)
        load(load_from, trainer=trainer, engine=engine)
        optimizer.param_groups[0]['capturable'] = True

    # Run the training engine for the requested number of epochs
    engine.run(train_loader, max_epochs=params["max_epochs"])
