import torch
from torch import nn
from torch.nn import functional as F

class CNN(nn.Module):
    def __init__(self, length: int, stride: int, channels: int, class_count: int, dropout: float):
        super().__init__()
        self.class_count = class_count
        embeddings = 34950
        self.dropout = nn.Dropout(p=dropout)
        self.stride_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=1,
            kernel_size=length,
            stride=stride,
        )
        self.initialise_layer(self.stride_conv)
        self.l_normstride = nn.LayerNorm([self.stride_conv.out_channels, embeddings//length])
        self.b_normstride = nn.BatchNorm1d(num_features=self.stride_conv.out_channels)
        self.conv1 = nn.Conv1d(
            in_channels=self.stride_conv.out_channels,
            out_channels=32,
            kernel_size=8,
            padding="same",
            stride=1,
        )
        self.initialise_layer(self.conv1)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=1)
        self.l_norm1 = nn.LayerNorm([self.conv1.out_channels, embeddings//length])
        self.b_norm1 = nn.BatchNorm1d(num_features=self.conv1.out_channels)
        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=32,
            kernel_size=8,
            padding="same",
            stride=1,
        )
        self.initialise_layer(self.conv2)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=1)
        self.l_norm2 = nn.LayerNorm([self.conv2.out_channels, embeddings//length - 3])
        self.b_norm2 = nn.BatchNorm1d(num_features=self.conv2.out_channels)
        self.fc1 = nn.Linear(4160, 100)
        self.initialise_layer(self.fc1)
        self.l_norm3 = nn.LayerNorm([100])
        self.norm3 = nn.BatchNorm1d(num_features=100)
        self.fc2 = nn.Linear(100, self.class_count)
        self.initialise_layer(self.fc2)

        self.convolution = nn.Sequential(
            self.stride_conv,
            nn.ReLU(),      
            self.conv1,
            nn.ReLU(),
            self.pool1, 
            self.conv2,
            nn.ReLU(),
            self.pool2,    
        )
        self.dense = nn.Sequential(
            self.fc1,
            nn.ReLU(),        
            self.fc2,
            nn.Sigmoid(),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = audio
        x = torch.flatten(x, start_dim = 0, end_dim=1)
        mean = torch.mean(x, dim=0)
        std = torch.std(x, dim=0) 
        # print(mean)
        # print(std)
        x = (x - mean) / std
        # x = F.relu(self.stride_conv(x))
        # x = F.relu(self.conv1(x))
        # x = self.pool1(x)
        # x = F.relu(self.conv2(x))
        # x = self.pool2(x)
        x = self.convolution(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.dense(x)
        # x = F.relu(self.fc1(self.dropout(x)))
        # x = nn.Sigmoid(self.fc2(self.dropout(x)))
        x = torch.reshape(x, (audio.size()[0], 10, -1))
        x = torch.mean(x, dim=1)
        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)
