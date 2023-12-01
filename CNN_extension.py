import torch
from torch import nn
from torch.nn import functional as F

class Conv1d(nn.Conv1d):
    def __init__(self, in_chan, out_chan, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_chan, out_chan, kernel_size, stride, 
                         padding, dilation, groups, bias)
        print("with weight standardisation")
    def forward(self, x):
            weight = self.weight
            weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                    keepdim=True)
            weight = weight - weight_mean
            std = weight.view(weight.size(0), -1).std(dim=1).view(-1,1,1)+1e-5
            weight = weight / std.expand_as(weight)
            return F.conv1d(x, weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

class EstBN(nn.Module):

    def __init__(self, num_features):
        super(EstBN, self).__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.register_buffer('estbn_moving_speed', torch.zeros(1))

    def forward(self, inp):
        ms = self.estbn_moving_speed.item()
        if self.training:
            with torch.no_grad():
                inp_t = inp.transpose(0, 1).contiguous().view(self.num_features, -1)
                running_mean = inp_t.mean(dim=1)
                inp_t = inp_t - self.running_mean.view(-1, 1)
                running_var = torch.mean(inp_t * inp_t, dim=1)
                self.running_mean.data.mul_(1 - ms).add_(ms * running_mean.data)
                self.running_var.data.mul_(1 - ms).add_(ms * running_var.data)
        out = inp - self.running_mean.view(1, -1, 1, 1)
        out = out / torch.sqrt(self.running_var + 1e-5).view(1, -1, 1, 1)
        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)
        out = weight * out + bias
        return out

class BCNorm(nn.Module):

    def __init__(self, num_channels, num_groups, eps, estimate=False):
        super(BCNorm, self).__init__()
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_groups, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_groups, 1))
        if estimate:
            self.bn = EstBN(num_channels)
        else:
            self.bn = nn.BatchNorm1d(num_channels)

    def forward(self, inp):
        out = self.bn(inp)
        out = out.view(1, inp.size(0) * self.num_groups, -1)
        out = torch.batch_norm(out, None, None, None, None, True, 0, self.eps, True)
        out = out.view(inp.size(0), self.num_groups, -1)
        out = self.weight * out + self.bias
        out = out.view_as(inp)
        return out

class CNN_extension(nn.Module):
    def __init__(self, length: int, stride: int, channels: int, class_count: int, minval: int, maxval: int, normalisation, out_channels: int, dropout: int, inner_norm):
        super().__init__()
        self.name = "CNN_extension1"
        self.kwargs = {'length': length, "stride": stride, "channels": channels, "class_count": class_count, "minval": minval, "maxval": maxval, "normalisation": normalisation, "out_channels": out_channels, "dropout": dropout, "inner_norm": inner_norm}
        self.minval = minval
        self.maxval = maxval
        self.normalisation = normalisation
        self.class_count = class_count
        embeddings = 34950
        padding_conv = 4
        stride_conv_output = int((embeddings - length)//stride + 1)
        conv1_output = (stride_conv_output - 8 + padding_conv*2) + 1
        pool1_output = (conv1_output - 4) + 1
        conv2_output = (pool1_output - 8 + padding_conv*2) + 1
        pool2_output = (conv2_output - 4) + 1
        self.stride_conv = nn.Conv1d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=length,
            stride=stride,
        )
        self.initialise_layer(self.stride_conv)
        self.conv1 = nn.Conv1d(
            in_channels=self.stride_conv.out_channels,
            out_channels=32,
            kernel_size=8,
            padding=padding_conv,
            stride=1,
        )
        self.initialise_layer(self.conv1)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=1)
        self.conv2 = nn.Conv1d(
            in_channels=self.conv1.out_channels,
            out_channels=32,
            kernel_size=8,
            padding=padding_conv,
            stride=1,
        )
        self.initialise_layer(self.conv2)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=1)
        self.fc1 = nn.Linear(pool2_output*self.conv2.out_channels, 100)
        self.initialise_layer(self.fc1)
        self.fc2 = nn.Linear(100, self.class_count)
        self.initialise_layer(self.fc2)
        self.dropout = nn.Dropout(dropout)

        if inner_norm == 'None':
            self.normstride = nn.Identity()
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
            self.norm3 = nn.Identity()
        elif inner_norm == "Layer":
            self.normstride = nn.LayerNorm([self.stride_conv.out_channels, stride_conv_output])
            self.norm1 = nn.LayerNorm([self.conv1.out_channels, conv1_output])
            self.norm2 = nn.LayerNorm([self.conv2.out_channels, conv2_output])
            self.norm3 = nn.LayerNorm([100])
        elif inner_norm == "Batch":
            self.normstride = nn.BatchNorm1d(num_features=self.stride_conv.out_channels)
            self.norm1 = nn.BatchNorm1d(num_features=self.conv1.out_channels)
            self.norm2 = nn.BatchNorm1d(num_features=self.conv2.out_channels)
            self.norm3 = nn.BatchNorm1d(num_features=100)
        elif inner_norm == "Group":
            self.normstride = nn.GroupNorm(num_groups=8, num_channels=self.stride_conv.out_channels)
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=self.conv1.out_channels)
            self.norm2 = nn.GroupNorm(num_groups=8, num_channels=self.conv2.out_channels)
            self.norm3 = nn.GroupNorm(num_groups=10, num_channels=100)
        elif inner_norm == "Batch-Channel":
            e = False
            self.normstride = BCNorm(num_groups=8, num_channels=self.stride_conv.out_channels, eps=1e-05, estimate=e)
            self.norm1 = BCNorm(num_groups=8, num_channels=self.conv1.out_channels, eps=1e-05, estimate=e)
            self.norm2 = BCNorm(num_groups=8, num_channels=self.conv2.out_channels, eps=1e-05, estimate=e)
            self.norm3 = BCNorm(num_groups=10, num_channels=100, eps=1e-05, estimate=e)
            print(f"estimate = {e}")

        self.convolution = nn.Sequential(
            self.stride_conv,
            self.normstride,
            nn.ReLU(),      
            self.conv1,
            self.norm1,
            nn.ReLU(),
            self.pool1, 
            self.conv2,
            self.norm2,
            nn.ReLU(),
            self.pool2,    
        )
        self.dense = nn.Sequential(
            self.dropout,
            self.fc1,
            self.norm3,
            nn.ReLU(),
            self.dropout,        
            self.fc2,
            nn.Sigmoid(),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = audio
        if self.normalisation == "minmax":
            x = (x - self.minval) / (self.maxval - self.minval) * 2 - 1

        x = torch.flatten(x, start_dim = 0, end_dim=1)

        if self.normalisation == "standardisation":
            mean = torch.mean(x, dim=0)
            std = torch.std(x, dim=0)
            x = (x - mean) / std

        x = self.convolution(x)
        x = torch.flatten(x, start_dim = 1)
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
