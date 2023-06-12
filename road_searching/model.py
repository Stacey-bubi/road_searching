import torch
import torch.nn as nn
from segmentation_models_pytorch import DeepLabV3Plus


class SegmentationModel(nn.Module):
    def __init__(self, encoder_name, encoder_depth, encoder_weights, encoder_output_stride, in_channels, classes,
                 activation, upsampling):
        super(SegmentationModel, self).__init__()
        self.model = DeepLabV3Plus(encoder_name=encoder_name, encoder_depth=encoder_depth,
                                   encoder_weights=encoder_weights, encoder_output_stride=encoder_output_stride,
                                   in_channels=in_channels, classes=classes, activation=activation,
                                   upsampling=upsampling)

    def forward(self, x):
        return self.model(x)
