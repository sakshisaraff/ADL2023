import time
from pathlib import Path

import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import evaluation

class Trainer:
    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            inter_eval_loader: DataLoader,
            criterion: nn.Module,
            optimizer: Optimizer,
            scheduler: torch.optim.lr_scheduler,
            summary_writer: SummaryWriter,
            device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.inter_eval_loader = inter_eval_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
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

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, loss, data_load_time, step_time)
                self.step += 1
                data_load_start_time = time.time()
            self.scheduler.step()

            self.summary_writer.add_scalar("epoch", epoch, self.step)
            if ((epoch + 1) % val_frequency) == 0:
                self.evaluate(self.inter_eval_loader)
                self.model.train()

    def print_metrics(self, epoch, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
            f"epoch: [{epoch}], "
            f"step: [{epoch_step}/{len(self.train_loader)}], "
            f"batch loss: {loss:.5f}, "
            f"data load time: "
            f"{data_load_time:.5f}, "
            f"step time: {step_time:.5f}"
        )

    def log_metrics(self, epoch, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
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

    """""
    The purpose of this function is to evaluate our model at regular intervals during training.
    This allows us to check in on how our training is going.
    """
    def evaluate(self, eval_loader: DataLoader):
        results = {"preds": []}
        total_loss = 0
        self.model.eval()
        with torch.no_grad():
            for filename, batch, labels in eval_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                results["preds"].extend(list(logits))

        auc = evaluation.evaluate(results["preds"], Path("../annotations/test_labels.pkl"))
        average_loss = total_loss / len(eval_loader)

        self.summary_writer.add_scalars(
            "area under the curve",
            {"test": auc},
            self.step
        )
        self.summary_writer.add_scalars(
            "loss",
            {"test": average_loss},
            self.step
        )
        print(f"validation loss: {average_loss:.5f}, auc: {auc * 100:2.2f}")