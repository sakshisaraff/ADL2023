import sys

import torch
from torch import nn
from torch.nn import functional as F

class SampleCNN(nn.Module):
    def __init__(self, class_count: int):
        super().__init__()
        self.name = "Deep"
        self.class_count = class_count
        self.kwargs = {"class_count": class_count}

        groups = 8
        self.conv0 = nn.Conv1d(
            in_channels=1,
            out_channels=128,
            kernel_size=3,
            stride=3,
        )
        self.initialise_layer(self.conv0)
        #self.norm0 = nn.BatchNorm1d(num_features=128)
        self.norm0 = nn.GroupNorm(num_groups=groups, num_channels=self.conv0.out_channels)

        self.conv1 = nn.Conv1d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding='same'
        )
        self.initialise_layer(self.conv1)
        #self.norm1 = nn.BatchNorm1d(num_features=128)
        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=self.conv1.out_channels)
        self.pool1 = nn.MaxPool1d(
            kernel_size=3,
            stride=1
        )

        self.conv2 = nn.Conv1d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding='same'
        )
        self.initialise_layer(self.conv2)
        #self.norm2 = nn.BatchNorm1d(num_features=128)
        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=self.conv2.out_channels)
        
        self.pool2 = nn.MaxPool1d(
            kernel_size=3,
            stride=3
        )

        self.conv3 = nn.Conv1d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding='same'
        )
        self.initialise_layer(self.conv3)
        #self.norm3 = nn.BatchNorm1d(num_features=256)
        self.norm3 = nn.GroupNorm(num_groups=groups, num_channels=self.conv3.out_channels)
        self.pool3 = nn.MaxPool1d(
            kernel_size=3,
            stride=3
        )

        self.conv4 = nn.Conv1d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding='same'
        )
        self.initialise_layer(self.conv4)
        # self.norm4 = nn.BatchNorm1d(num_features=256)
        self.norm4 = nn.GroupNorm(num_groups=groups, num_channels=self.conv4.out_channels)
        self.pool4 = nn.MaxPool1d(
            kernel_size=3,
            stride=3
        )

        self.conv5 = nn.Conv1d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding='same'
        )
        self.initialise_layer(self.conv5)
        # self.norm5 = nn.BatchNorm1d(num_features=256)
        self.norm5 = nn.GroupNorm(num_groups=groups, num_channels=self.conv5.out_channels)
        self.pool5 = nn.MaxPool1d(
            kernel_size=3,
            stride=3
        )

        self.conv6 = nn.Conv1d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding='same'
        )
        self.initialise_layer(self.conv6)
        # self.norm6 = nn.BatchNorm1d(num_features=256)
        self.norm6 = nn.GroupNorm(num_groups=groups, num_channels=self.conv6.out_channels)
        self.pool6 = nn.MaxPool1d(
            kernel_size=3,
            stride=3
        )

        self.conv7 = nn.Conv1d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding='same'
        )
        self.initialise_layer(self.conv7)
        # self.norm7 = nn.BatchNorm1d(num_features=256)
        self.norm7 = nn.GroupNorm(num_groups=groups, num_channels=self.conv7.out_channels)
        self.pool7 = nn.MaxPool1d(
            kernel_size=3,
            stride=3
        )

        self.conv8 = nn.Conv1d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding='same'
        )
        self.initialise_layer(self.conv8)
        # self.norm8 = nn.BatchNorm1d(num_features=256)
        self.norm8 = nn.GroupNorm(num_groups=groups, num_channels=self.conv8.out_channels)
        self.pool8 = nn.MaxPool1d(
            kernel_size=3,
            stride=3
        )

        self.conv9 = nn.Conv1d(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding='same'
        )
        self.initialise_layer(self.conv9)
        # self.norm9 = nn.BatchNorm1d(num_features=512)
        self.norm9 = nn.GroupNorm(num_groups=groups, num_channels=self.conv9.out_channels)
        self.pool9 = nn.MaxPool1d(
            kernel_size=3,
            stride=3
        )

        # Effectively a fully connected layer.
        self.conv10 = nn.Conv1d(
            in_channels=512,
            out_channels=512,
            kernel_size=1,
            stride=1,
            padding='same'
        )
        self.initialise_layer(self.conv10)
        # self.norm10 = nn.BatchNorm1d(num_features=512)
        self.norm10 = nn.GroupNorm(num_groups=groups, num_channels=self.conv10.out_channels)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(512, 50)

        self.convolution = nn.Sequential(
            self.conv0,
            self.norm0,
            nn.ReLU(),

            self.conv1,
            self.norm1,
            nn.ReLU(),
            self.pool1,

            self.conv2,
            self.norm2,
            nn.ReLU(),
            self.pool2,

            self.conv3,
            self.norm3,
            nn.ReLU(),
            self.pool3,

            self.conv4,
            self.norm4,
            nn.ReLU(),
            self.pool4,

            self.conv5,
            self.norm5,
            nn.ReLU(),
            self.pool5,

            self.conv6,
            self.norm6,
            nn.ReLU(),
            self.pool6,

            self.conv7,
            self.norm7,
            nn.ReLU(),
            self.pool7,

            self.conv8,
            self.norm8,
            nn.ReLU(),
            self.pool8,

            self.conv9,
            self.norm9,
            nn.ReLU(),
            self.pool9,

            self.conv10,
            self.norm10,
            nn.ReLU(),
        )

        self.dense = nn.Sequential(
            self.dropout,
            self.fc,
            nn.Sigmoid(),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(audio, start_dim=0, end_dim=1)
        x = self.convolution(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dense(x)
        x = torch.reshape(x, (audio.size()[0], 10, -1))
        x = torch.mean(x, dim=1)
        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)
