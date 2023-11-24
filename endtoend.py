#!/usr/bin/env python3

import argparse
import io
from pathlib import Path
from multiprocessing import cpu_count

import torch
import torch.backends.cudnn
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import dataset
from Trainer import Trainer
from CNN import CNN

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

torch.backends.cudnn.benchmark = True

default_dataset_dir = Path.home() / ".cache" / "torch" / "datasets"

# region Argument Parsing

parser = argparse.ArgumentParser(
    description="Train a simple CNN on MagnaTagATune for End-to-End Learning",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--dataset-root", default=default_dataset_dir)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--learning-rate", default=1e-3, type=float, help="Learning rate")
parser.add_argument("--dropout", default=0, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--normalisation", default="Sakshi", type=str)
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
    "--eval-frequency",
    default=1,
    type=int,
    help="How frequently to evaluate the model in number of epochs",
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
parser.add_argument(
    "--mode",
    default="training",
    help="Choices are hyperparameter-tuning, or training",
)

# endregion

def main(args):
    # region Data Loading
    path_annotations_train = Path("annotations/train_labels.pkl")
    path_annotations_val = Path("annotations/val_labels.pkl")
    path_annotations_test = Path("annotations/test_labels.pkl")
    path_samples_train = Path("samples/train")
    path_samples_val = Path("samples/val")
    path_samples_test = Path("samples/test")
    train_dataset = dataset.MagnaTagATune(path_annotations_train, path_samples_train)
    val_dataset = dataset.MagnaTagATune(path_annotations_val, path_samples_val)
    test_dataset = dataset.MagnaTagATune(path_annotations_test, path_samples_test)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=args.batch_size,
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
    # endregion

    # Values are the same for all datasets, hence we don't recalculate them for different contexts.
    minval, maxval = minmax(train_loader)

    trainer = None
    hyperparameter_choices = {
        "batch_size": args.batch_size,
        "epochs": 1,
        "learning_rate": args.learning_rate,
        "momentum": args.momentum,
        "dropout": args.dropout,
        "length_conv": args.length_conv,
        "stride_conv": args.stride_conv,
        "normalisation": args.normalisation
    }
    """
    If you specify on the command line that you want to tune hyperparameters,
    use grid search to alter hyperparameter_choices.
    """
    if args.mode == "hyperparameter-tuning":
        scored_hyperparameter_choices = []
        hyperparameter_possibilities = {
            #"batch_size": [],
            "learning_rate": [0.001, 0.005],
            #"normalisation": ["minmax", "Sakshi"]
            #"learning_rate": [0.0005, 0.001, 0.0015, 0.005, 0.01],
            #'"momentum": [0.1, 0.9, 0.92, 0.94, 0.97, 0.99],
            #"normalisation": ["minmax", "Sakshi"]
            #"epochs": [],
            #"dropout": [],
            #"length_conv": [],
            #"stride_conv": []
        }
        for hyperparameter, possibilities in hyperparameter_possibilities.items():
            best_auc = 0
            best_choice = None
            for possibility in possibilities:
                if hyperparameter == "batch_size":
                    train_loader = torch.utils.data.DataLoader(
                        train_dataset,
                        shuffle=True,
                        batch_size=possibility,
                        pin_memory=True,
                        num_workers=args.worker_count,
                    )
                hyperparameter_choices[hyperparameter] = possibility
                trainer, model = train(
                    args,
                    minval,
                    maxval,
                    train_loader,
                    val_loader,
                    args.eval_frequency,
                    args.print_frequency,
                    args.log_frequency,
                    hyperparameter_choices,
                    path_annotations_val,
                )
                auc = trainer.evaluate(path_annotations_val, val_loader)
                scored_hyperparameter_choices.append((auc, hyperparameter_choices.copy()))
                if auc > best_auc:
                    best_choice = possibility
                    best_auc = auc

            """
            Once we've tested all the different batch sizes,
            we can make a loader with the best one.
            """
            if hyperparameter == "batch_size":
                train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    shuffle=True,
                    batch_size=hyperparameter_choices["batch_size"],
                    pin_memory=True,
                    num_workers=args.worker_count,
                )
            hyperparameter_choices[hyperparameter] = best_choice
        print("Hyperparameter tuning done")
        print(scored_hyperparameter_choices)
    else:
        trainer, model = train(
            args,
            minval,
            maxval,
            train_loader,
            val_loader,
            args.eval_frequency,
            args.print_frequency,
            args.log_frequency,
            hyperparameter_choices,
            path_annotations_val
        )

    print("Evaluating against test data...")
    print(hyperparameter_choices)
    print("Test Results:")
    trainer.evaluate(path_annotations_test, test_loader)

def get_summary_writer_log_dir(args: argparse.Namespace, hyperparameters) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """

    tb_log_dir_prefix = io.StringIO()
    tb_log_dir_prefix.write(f'CNN_')
    for hyperparameter, value in hyperparameters.items():
        tb_log_dir_prefix.write(f'{hyperparameter}={value}_')
    tb_log_dir_prefix.write(f'run_')
    tb_log_dir_prefix = tb_log_dir_prefix.getvalue()
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)

def train(
        args,
        minval,
        maxval,
        train_loader,
        inter_eval_loader,
        eval_frequency,
        print_frequency,
        log_frequency,
        hyperparameters,
        sample_path
):
    log_dir = get_summary_writer_log_dir(args, hyperparameters)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
        str(log_dir),
        flush_secs=5
    )
    print("Start of Training")
    print("Hyperparameters: " + str(hyperparameters))
    model = CNN(length=args.length_conv, stride=args.stride_conv, channels=1, class_count=50, dropout=args.dropout, minval=minval, maxval=maxval, normalisation=hyperparameters["normalisation"])
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=hyperparameters["learning_rate"], momentum=hyperparameters["momentum"])
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    trainer = Trainer(
        model,
        train_loader,
        inter_eval_loader,
        criterion,
        optimizer,
        summary_writer,
        DEVICE
    )
    trainer.train(
        sample_path,
        hyperparameters["epochs"],
        eval_frequency,
        print_frequency=print_frequency,
        log_frequency=log_frequency,
    )
    summary_writer.close()
    return trainer, model

def minmax(dataloader):
    maxval = float("-inf")
    minval = float("inf")
    for filename, data, label in dataloader:
        if minval > torch.min(data):
            minval = torch.min(data)
        if maxval < torch.max(data):
            maxval = torch.max(data)
    return minval, maxval

if __name__ == "__main__":
    main(parser.parse_args())
