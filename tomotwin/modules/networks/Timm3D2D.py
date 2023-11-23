import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from tomotwin.modules.networks.torchmodel import TorchModel


class Timm2D3D(TorchModel):
    class Model(nn.Module):
        def __init__(
                self,
                modelname: str,
                norm: nn.Module,
        ):
            super().__init__()
            self.model = timm.create_model(modelname, pretrained=True)
            self.conv_3d_1 = nn.Conv3d(1, 64, 3)
            self.conv_3d_2 = nn.Conv3d(64, 64, 3)
            self.conv_3d_3 = nn.Conv3d(64, 32, 3)
            self.conv_3d_4 = nn.Conv3d(32, 1, 3)
            self.max_pooling = nn.MaxPool3d((2, 2, 2))
            self.relu = nn.LeakyReLU()
            self.norm_1_2 = nn.GroupNorm(num_channels=64, num_groups=64)
            self.norm_2_3 = nn.GroupNorm(num_channels=64, num_groups=64)
            self.norm_3_4 = nn.GroupNorm(num_channels=32, num_groups=32)
            self.pool = nn.AdaptiveAvgPool3d((32))
            self.conv2d = nn.Conv2d(32, 3, 3)
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
            x = F.pad(inputtensor, (1, 2, 1, 2, 1, 2))
            # print("Shape input pad", inputtensor.shape)
            x = self.conv_3d_1(inputtensor)
            if self.norm_1_2 is not None:
                x = self.norm_1_2(x)
                x = self.relu(x)
            x = self.conv_3d_2(x)
            x = self.max_pooling(x)
            if self.norm_2_3 is not None:
                x = self.norm_2_3(x)
                x = self.relu(x)
            x = self.conv_3d_3(x)
            if self.norm_3_4 is not None:
                x = self.norm_3_4(x)
                x = self.relu(x)
            x = self.conv_3d_4(x)
            # print("x after conv3d", x.shape)
            x = x.squeeze(1)
            # print("x after squeeze", x.shape)
            x = self.pool(x)
            x = self.conv2d(x)
            # print("x after conv2d", x.shape)
            x = self.model(x)
            # print("out model", x.shape)
            x = x.reshape(x.size(0), -1)  # flatten
            #print("reshape size", x.shape)
            x = self.headnet(x)
            x = F.normalize(x, p=2, dim=1)
            # print("headnet size", x.shape)
            return x

    def __init__(
            self,
    ):
        self.model = self.Model('efficientnet_b3',
                                norm=nn.GroupNorm)

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
