import torch
from torch import nn
import torchaudio.transforms as T

class CNNspectogram(nn.Module):
    def __init__(self, length: int, stride: int, channels: int, class_count: int, minval: int, maxval: int, normalisation, out_channels: int, n_mels=128, sample_rate=16000):
        super().__init__()
        self.kwargs = {'length': length, "stride": stride, "channels": channels, "class_count": class_count, "minval": minval, "maxval": maxval, "normalisation": normalisation, "out_channels": out_channels}
        self.minval = minval
        self.maxval = maxval
        self.normalisation = normalisation
        self.class_count = class_count

        # MelSpectrogram layer
        self.mel_spectrogram = T.MelSpectrogram(sample_rate=12000, n_fft=1024, hop_length=512, n_mels=128)
        self.C = 10000  # Compression constant

        # Assuming the dimensions of the mel spectrogram for the CNN layers
        mel_bins = n_mels
        #value representing the length of the time dimension in the Mel spectrogram. It's determined based on the size of your input audio data and the parameters of the Mel spectrogram transformation
        time_steps = 34950 // 512  # Example calculation, adjust as needed
        padding_conv = 4
        # CNN layers
        conv1_output = (time_steps - 8 + padding_conv*2) + 1
        pool1_output = (conv1_output - 4) + 1
        conv2_output = (pool1_output - 8 + padding_conv*2) + 1
        pool2_output = (conv2_output - 4) + 1

        self.conv1 = nn.Conv1d(
            in_channels=mel_bins,
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
        self.convolution = nn.Sequential(   
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
        # Convert audio to Mel spectrogram
        x = audio
        mel = self.mel_spectrogram(x)

        # Apply dynamic range compression
        x = torch.log1p(self.C * mel)

        # Normalize if required
        if self.normalisation == "minmax":
            x = (x - self.minval) / (self.maxval - self.minval) * 2 - 1

        x = torch.flatten(x, start_dim = 0, end_dim=1)

        if self.normalisation == "Sakshi":
            mean = torch.mean(x, dim=0)
            std = torch.std(x, dim=0)
            x = (x - mean) / std
            
        # Apply CNN layers
        x = self.convolution(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.dense(x)
        x = torch.reshape(x, (audio.size()[0], 10, -1))
        x = torch.mean(x, dim=1)
        return x

        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)

# Example usage
# cnn = CNN(length=5, stride=2, channels=1, class_count=10, minval=0, maxval=1, normalisation="minmax", out_channels=16)
        mel_spectrogram = self.mel_spectrogram_transform(samples)
        self.mel_spectrogram_transform = transforms.MelSpectrogram(sample_rate=12000, n_fft=1024, hop_length=512, n_mels=128)
