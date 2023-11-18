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
    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
        str(log_dir),
        flush_secs=5
    )

    # region Data Loading
    path_annotations_train = Path("../annotations/train_labels.pkl")
    path_annotations_val = Path("../annotations/val_labels.pkl")
    path_annotations_test = Path("../annotations/test_labels.pkl")
    path_samples_train = Path("../samples/train")
    path_samples_val = Path("../samples/val")
    path_samples_test = Path("../samples/test")
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

    hyperparameter_choices = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate
    }

    """
    If you specify on the command line that you want to tune hyperparameters,
    use grid search to alter hyperparameter_choices
    """
    if args.mode == "hyperparameter-tuning":
        hyperparameter_possibilities = {
            "batch_size": [8, 16, 32, 64, 128, 256, 512],
            "epochs": [5, 10, 20, 40],
            "learning_rate": [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
        }

        for hyperparameter, possibilities in hyperparameter_possibilities:
            best_auc = 0
            for possibility in possibilities:
                if hyperparameter == "batch_size":
                    train_loader = torch.utils.data.DataLoader(
                        train_dataset,
                        shuffle=True,
                        batch_size=possibility,
                        pin_memory=True,
                        num_workers=args.worker_count,
                    )
                trainer, model = train(
                    train_loader,
                    val_loader,
                    summary_writer,
                    args.eval_frequency,
                    args.print_frequency,
                    args.log_frequency,
                    hyperparameter_choices
                )
                auc = trainer.evaluate(val_loader)
                if auc > best_auc:
                    hyperparameter_choices[hyperparameter] = possibility
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

    trainer, model = train(
        train_loader,
        val_loader,
        summary_writer,
        args.eval_frequency,
        args.print_frequency,
        args.log_frequency,
        hyperparameter_choices
    )
    print("Test Results:")
    trainer.evaluate(test_loader)

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
    tb_log_dir_prefix = f'CNN_bs={args.batch_size}_lr={args.learning_rate}_with_scheduler_run_'
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)

def train(
        train_loader,
        inter_eval_loader,
        summary_writer,
        eval_frequency,
        print_frequency,
        log_frequency,
        hyperparameters
):
    model = CNN(channels=1, class_count=50)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    trainer = Trainer(
        model,
        train_loader,
        inter_eval_loader,
        criterion,
        optimizer,
        scheduler,
        summary_writer,
        DEVICE
    )
    trainer.train(
        hyperparameters["epochs"],
        eval_frequency,
        print_frequency=print_frequency,
        log_frequency=log_frequency,
    )

    return trainer, model


if __name__ == "__main__":
    main(parser.parse_args())