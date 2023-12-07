import sys

import torch
from torch import nn
from torch.nn import functional as F

# class Conv1d(nn.Conv1d):
#     def __init__(self, in_chan, out_chan, kernel_size, stride=1, 
#                  padding=0, dilation=1, groups=1, bias=True):
#         super().__init__(in_chan, out_chan, kernel_size, stride, 
#                          padding, dilation, groups, bias)
#         print("with weight standardisation")
#     def forward(self, x):
#             weight = self.weight
#             weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
#                                     keepdim=True)
#             weight = weight - weight_mean
#             std = weight.view(weight.size(0), -1).std(dim=1).view(-1,1,1)+1e-5
#             weight = weight / std.expand_as(weight)
#             return F.conv1d(x, weight, self.bias, self.stride,
#                             self.padding, self.dilation, self.groups)

class SampleCNN(nn.Module):
    def __init__(self, class_count: int):
        super().__init__()
        self.name = "Deep"
        self.class_count = class_count
        self.kwargs = {"class_count": class_count}

        self.conv0 = nn.Conv1d(
            in_channels=1,
            out_channels=128,
            kernel_size=3,
            stride=3,
        )
        self.initialise_layer(self.conv0)
        self.norm0 = nn.BatchNorm1d(num_features=128)

        self.conv1 = nn.Conv1d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding='same'
        )
        self.initialise_layer(self.conv1)
        self.norm1 = nn.BatchNorm1d(num_features=128)
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
        self.norm2 = nn.BatchNorm1d(num_features=128)
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
        self.norm3 = nn.BatchNorm1d(num_features=256)
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
        self.norm4 = nn.BatchNorm1d(num_features=256)
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
        self.norm5 = nn.BatchNorm1d(num_features=256)
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
        self.norm6 = nn.BatchNorm1d(num_features=256)
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
        self.norm7 = nn.BatchNorm1d(num_features=256)
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
        self.norm8 = nn.BatchNorm1d(num_features=256)
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
        self.norm9 = nn.BatchNorm1d(num_features=512)
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
        self.norm10 = nn.BatchNorm1d(num_features=512)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, 256)
        self.norm11 = nn.BatchNorm1d(num_features=256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 50)

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
            self.norm11,
            nn.ReLU(),
            self.dropout2,
            self.fc2,
            nn.Sigmoid(),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # catted_clips = []
        # for clip in audio:
        #     catted_clip = torch.cat((clip[0], clip[1], clip[2], clip[3], clip[4], clip[5], clip[6], clip[7], clip[8], clip[9]))
        #     catted_clips.append(catted_clip)
        # catted_clips = torch.tensor(catted_clips)
        # print(catted_clips.size())
        # sys.exit()

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
