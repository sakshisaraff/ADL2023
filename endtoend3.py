#!/usr/bin/env python3

import argparse
from pathlib import Path
from multiprocessing import cpu_count

import torch
import torch.backends.cudnn
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import dataset
import Trainer
import CNN


torch.backends.cudnn.benchmark = True

default_dataset_dir = Path.home() / ".cache" / "torch" / "datasets"

parser = argparse.ArgumentParser(
    description="Train a simple CNN on MagnaTagATune for End-to-End Learning",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--dataset-root", default=default_dataset_dir)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--learning-rate", default=1e-3, type=float, help="Learning rate")
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
    default=1,
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
    path_annotations_train = Path("../annotations/train_labels.pkl")
    path_samples_train = Path("../samples/train")
    train_dataset = dataset.MagnaTagATune(path_annotations_train, path_samples_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count,
    )

    model = CNN(length=args.length_conv, stride=args.stride_conv, channels=1, class_count=50)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )

    trainer = Trainer(
        model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, summary_writer, DEVICE
    )
    trainer.train(
        args.epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
    )

    summary_writer.close()

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
    tb_log_dir_prefix = f'CNN_bs={args.batch_size}_lr={args.learning_rate}_with_scheduler_length={args.length_conv}_stride={args.stride_conv}_run_'
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)


if __name__ == "__main__":
    main(parser.parse_args())