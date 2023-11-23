import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from tomotwin.modules.networks.torchmodel import TorchModel


class Timm2D3D(TorchModel):
    class Model(nn.Module):
        def __init__(
                self,
                modelname: str
        ):
            super().__init__()
            self.model = timm.create_model(modelname, pretrained=False)
            self.conv_3d = nn.Conv3d(1, 1, 3)
            self.pool = nn.AdaptiveAvgPool3d((144))
            self.conv2d = nn.Conv2d(144, 3, 3)
            self.headnet = self._make_headnet(
                1000, 2048, 32, dropout=0
            )

        @staticmethod
        def _make_headnet(
                in_c1: int, out_c1: int, out_head: int, dropout: float
        ) -> nn.Sequential:
            headnet = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_c1, out_c1),
                nn.LeakyReLU(),
                nn.Linear(out_c1, out_c1),
                nn.LeakyReLU(),
                nn.Linear(out_c1, out_head),
            )
            return headnet

        def forward(self, inputtensor):
            """
            Forward pass through the network
            :param inputtensor: Input tensor
            """
            # print("Shape input", inputtensor.shape)
            inputtensor = F.pad(inputtensor, (1, 2, 1, 2, 1, 2))
            # print("Shape input pad", inputtensor.shape)
            x = self.conv_3d(inputtensor)
            # print("x after conv3d", x.shape)
            x = x.squeeze(1)
            # print("x after squeeze", x.shape)
            x = self.pool(x)
            # print("x after pool", x.shape)
            x = self.conv2d(x)
            # print("x after conv2d", x.shape)
            x = self.model(x)
            # print("out model", x.shape)
            x = x.reshape(x.size(0), -1)  # flatten
            # print("reshape size", x.shape)
            x = self.headnet(x)
            # print("headnet size", x.shape)
            return x

    def __init__(
            self,
    ):
        self.model = self.Model('efficientnet_b3')

    def init_weights(self):
        def _init_weights(model):
            if isinstance(model, nn.Conv3d):
                torch.nn.init.kaiming_normal_(model.weight)

        self.model.apply(_init_weights)

    def get_model(self) -> nn.Module:
        """
        Returns the model
        """
        return self.model
