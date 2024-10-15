
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import ResNet

LOGGER = logging.getLogger(__name__)


class MultimodalModel(torch.nn.Module):
    def __init__(self, image_model, spectrum_model):
        super(MultimodalModel, self).__init__()
        self.image_model = image_model
        self.spectrum_model = spectrum_model
                
    def forward(self, image, spectrum):
        out_1 = self.image_model(image)
        out_2 = self.spectrum_model(spectrum)

        return F.normalize(out_1, dim=-1), F.normalize(out_2, dim=-1)


def build_model(params):
    image_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=params[params["con_backbone"]]["pretrained"])
    image_model.fc = nn.Linear(2048, params["latent_dimensionality"])

    # Change the first layer to accept 5 bands
    if '5band' in params["dataset_file"]:
        image_model.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    spectrum_model = ResNet(params["latent_dimensionality"], **params[params["con_backbone"]])
    model = MultimodalModel(image_model, spectrum_model)

    num_of_parameters = sum(map(torch.numel, model.parameters()))
    LOGGER.info("%s trainable params: %d", params['con_backbone'], num_of_parameters)

    return model

    
