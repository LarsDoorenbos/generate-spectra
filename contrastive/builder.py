
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
    image_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
    image_model.fc = nn.Linear(2048, params["latent_dimensionality"])
    
    spectrum_model = ResNet(params["latent_dimensionality"], **params[params["con_backbone"]])
    
    model = MultimodalModel(image_model, spectrum_model)

    num_of_parameters = sum(map(torch.numel, model.parameters()))
    LOGGER.info("%s trainable params: %d", params['con_backbone'], num_of_parameters)

    return model

    
