import torch
from torch import nn
from torch.nn import functional as F

class CNN(nn.Module):
    def __init__(self, length: int, stride: int, channels: int, class_count: int):
        super().__init__()
        self.class_count = class_count
        self.stride_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=1,
            kernel_size=length,
            stride=stride,
        )
        self.initialise_layer(self.stride_conv)
        self.conv1 = nn.Conv1d(
            in_channels=self.stride_conv.out_channels,
            out_channels=32,
            kernel_size=8,
            padding="same",
            stride=1,
        )
        self.initialise_layer(self.conv1)
        #need stride 1, any padding??
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=1)
        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=32,
            kernel_size=8,
            padding="same",
            stride=1,
        )
        self.initialise_layer(self.conv2)
        #need stride 1, any padding??
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=1)
        #how to use variable for input value?
        self.fc1 = nn.Linear(4160, 100)
        self.initialise_layer(self.fc1)
        self.fc2 = nn.Linear(100, 50)
        self.initialise_layer(self.fc2)


    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = audio
        x = torch.flatten(x, start_dim = 0).reshape((10*audio.size()[0], 1, -1))
        x = F.relu(self.stride_conv(x))
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, start_dim = 1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.reshape(x, (audio.size()[0], 10, -1))
        x = torch.mean(x, dim=1)
        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)