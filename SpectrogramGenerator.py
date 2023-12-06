import argparse
from pathlib import Path
import pickle

import torch
import torch.backends.cudnn
from torch.utils.data import DataLoader
from torchaudio import transforms

import dataset

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

torch.backends.cudnn.benchmark = True

default_dataset_dir = Path.home() / ".cache" / "torch" / "datasets"

parser = argparse.ArgumentParser(
    description="Generates Mel-Spectrograms from Mono audio.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)


def main(args):
    path_annotations_train = Path("../adl_data/annotations/train_labels.pkl")
    path_annotations_val = Path("annotations/val_labels.pkl")
    path_annotations_test = Path("annotations/test_labels.pkl")
    train_dataset = dataset.MagnaTagATune(path_annotations_train, Path("../adl_data/samples/train"))
    val_dataset = dataset.MagnaTagATune(path_annotations_val, Path("../adl_data/samples/val"))
    test_dataset = dataset.MagnaTagATune(path_annotations_test, Path("../adl_data/samples/test"))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=1,
        pin_memory=True,
        num_workers=4,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
    )

    C = 10000
    transform = transforms.MelSpectrogram(sample_rate=12000, n_fft=1024, hop_length=512, n_mels=128)

    train_output = []
    for filename, batch, labels in train_loader:
        print(batch.size())
        spectrogram = transform(batch)
        print(spectrogram.size())
        spectrogram = torch.log1p(C * spectrogram)
        print(spectrogram.size())
        spectrogram = torch.flatten(spectrogram, start_dim=3)
        print(spectrogram.size())
        train_output.append(spectrogram)
    train_output = torch.cat(train_output, dim=0)
    print(train_output.size())

    with open(Path('spectrogram/train_spectrograms.pkl'), 'wb') as file:
        pickle.dump(train_output, file)

    test_output = []
    for filename, batch, labels in test_loader:
        print(batch.size())
        spectrogram = transform(batch)
        print(spectrogram.size())
        spectrogram = torch.log1p(C * spectrogram)
        print(spectrogram.size())
        spectrogram = torch.flatten(spectrogram, start_dim=3)
        print(spectrogram.size())
        test_output.append(spectrogram)
    test_output = torch.cat(test_output, dim=0)

    with open(Path('spectrogram/test_spectrograms.pkl'), 'wb') as file:
        pickle.dump(test_output, file)

    val_output = []
    for filename, batch, labels in val_loader:
        spectrogram = transform(batch)
        spectrogram = torch.log1p(C * spectrogram)
        spectrogram = torch.flatten(spectrogram, start_dim=3)
        val_output.append(spectrogram)
    val_output = torch.cat(val_output, dim=0)

    with open(Path('spectrogram/val_spectrograms.pkl'), 'wb') as file:
        pickle.dump(val_output, file)


if __name__ == "__main__":
    main(parser.parse_args())
