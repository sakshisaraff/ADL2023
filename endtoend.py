#!/usr/bin/env python3
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple

import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import dataset
import evaluation
import argparse
from pathlib import Path

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Train a simple CNN on MagnaTagATune for End-to-End Learning",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
default_dataset_dir = Path.home() / ".cache" / "torch" / "datasets"
parser.add_argument("--dataset-root", default=default_dataset_dir)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--learning-rate", default=1e-2, type=float, help="Learning rate")
parser.add_argument(
    "--length-conv",
    default=256,
    type=int,
    help="Length used in the Stride Convolution Layer",
)
parser.add_argument(
    "--stride-conv",
    default=256,
    type=int,
    help="Stride used in the Stride Convolution Layer",
)
parser.add_argument(
    "--batch-size",
    default=10,
    type=int,
    help="Number of audio examples within each batch",
)
parser.add_argument(
    "--epochs",
    default=20,
    type=int,
    help="Number of epochs (passes through the entire dataset) to train for",
)
parser.add_argument(
    "--val-frequency",
    default=2,
    type=int,
    help="How frequently to test the model on the validation set in number of epochs",
)
parser.add_argument(
    "--log-frequency",
    default=10,
    type=int,
    help="How frequently to save logs to tensorboard in number of steps",
)
parser.add_argument(
    "--print-frequency",
    default=10,
    type=int,
    help="How frequently to print progress to the command line in number of steps",
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=cpu_count(),
    type=int,
    help="Number of worker processes used to load data.",
)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def main(args):
    #args.dataset_root.mkdir(parents=True, exist_ok=True)
    path_annotations_train = Path("../annotations/train_labels.pkl")
    path_annotations_val = Path("../annotations/val_labels.pkl")
    path_annotations_test = Path("../annotations/test_labels.pkl")
    path_samples_train = Path("../samples/train")
    path_samples_val = Path("../samples/val")
    path_samples_test = Path("../samples/test")
    train_dataset = dataset.MagnaTagATune(path_annotations_train, path_samples_train)
    val_dataset = dataset.MagnaTagATune(path_annotations_val, path_samples_val)
    test_dataset = dataset.MagnaTagATune(path_annotations_test, path_samples_test)
    print(train_dataset.samples_path)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count,
    )
    # batch_size=args.batch_size,
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        num_workers=args.worker_count,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
    )

    model = CNN(length=args.length_conv, stride=args.stride_conv, channels=1, class_count=50)
    criterion = nn.BCELoss()
    #hyperparameter tune learning_rate
    #add momentum for improvement
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )
    trainer = Trainer(
        model, train_loader, val_loader, criterion, optimizer, summary_writer, DEVICE
    )
    trainer.train(
        args.epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
    )

    summary_writer.close()


class CNN(nn.Module):
    def __init__(self, length: int, stride: int, channels: int, class_count: int):
        super().__init__()
        self.class_count = class_count
        self.stride_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=1,
            kernel_size=length,
            padding=0,
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
        padding = 0
        self.initialise_layer(self.conv2)
        #need stride 1, any padding??
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=1)
        #how to use variable for input value?
        self.fc1 = nn.Linear(4160, 100) 
        self.initialise_layer(self.fc1)
        self.fc2 = nn.Linear(100, 50)
        self.initialise_layer(self.fc2)


    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        #print(audio.size())
        x = torch.mean(audio, dim=1)
        #print(x.size())
        x = F.relu(self.stride_conv(x))
        #print(x.size())
        x = F.relu(self.conv1(x))
        #print(x.size())
        x = self.pool1(x)
        #print(x.size())
        x = F.relu(self.conv2(x))
        #print(x.size())
        x = self.pool2(x)
        #print(x.size())
        x = torch.flatten(x, start_dim = 1)
        #print(x.size())
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        #print(x.size())
        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer: SummaryWriter,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0

    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0
    ):
        self.model.train()
        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()
            for filename, batch, labels in self.train_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                data_load_end_time = time.time()
                logits = self.model.forward(batch)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # with torch.no_grad():
                #     auc = evaluation.evaluate(logits, labels)
                #     print(auc)

                # data_load_time = data_load_end_time - data_load_start_time
                # step_time = time.time() - data_load_end_time
                # if ((self.step + 1) % log_frequency) == 0:
                #     self.log_metrics(epoch, accuracy, loss, data_load_time, step_time)
                # if ((self.step + 1) % print_frequency) == 0:
                #     self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                # self.step += 1
                # data_load_start_time = time.time()

            self.summary_writer.add_scalar("epoch", epoch, self.step)
            if ((epoch + 1) % val_frequency) == 0:
                self.validate()
                self.model.train()

    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"batch accuracy: {accuracy * 100:2.2f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}"
        )

    def log_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
                "accuracy",
                {"train": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
        self.summary_writer.add_scalar(
                "time/data", data_load_time, self.step
        )
        self.summary_writer.add_scalar(
                "time/data", step_time, self.step
        )

    def validate(self):
        #results = {"preds": [], "labels": []}
        results = []
        total_loss = 0
        self.model.eval()

        with torch.no_grad():
            for filename, batch, labels in self.val_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                # preds = logits.argmax(dim=-1).cpu().numpy()
                # results["preds"].extend(list(preds))
                # results["labels"].extend(list(labels.cpu().numpy()))
                auc = evaluation.evaluate(logits, Path("../annotations/val_labels.pkl"))
                results.append(auc)
        
        print(results)
        average_loss = total_loss / len(self.val_loader)

        # self.summary_writer.add_scalars(
        #         "accuracy",
        #         {"test": accuracy},
        #         self.step
        # )
        self.summary_writer.add_scalars(
                "loss",
                {"test": average_loss},
                self.step
        )
        print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")

def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = f'CNN_bs={args.batch_size}_lr={args.learning_rate}_length={args.length_conv}_stride={args.stride_conv}_run_'
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)


if __name__ == "__main__":
    main(parser.parse_args())