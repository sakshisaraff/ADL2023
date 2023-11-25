import time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import evaluation

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Trainer:
    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            inter_eval_loader: DataLoader,
            criterion: nn.Module,
            optimizer: Optimizer,
            summary_writer: SummaryWriter,
            device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.inter_eval_loader = inter_eval_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0

    def train(
            self,
            hyperparameter_choices,
            sample_path,
            model_path: Path,
            epochs: int,
            val_frequency: int,
            print_frequency: int = 20,
            log_frequency: int = 5,
            start_epoch: int = 0
    ):
        self.model.train()
        print(f'before: {count_parameters(self.model)}')
        best_auc = 0
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
            #self.scheduler.step()

            self.summary_writer.add_scalar("epoch", epoch, self.step)
            if ((epoch + 1) % val_frequency) == 0:
                auc = self.evaluate(sample_path, self.inter_eval_loader)
                if auc >= best_auc:
                    print("Saving model.")
                    torch.save({
                        'args': hyperparameter_choices,
                        'model': self.model.state_dict(),
                        'auc': auc,
                        'epoch': epoch
                    }, model_path)



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
        # self.summary_writer.add_scalars(
        #     "auc",
        #     {"train": float(auc.item())},
        #     self.step
        # )
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
    def evaluate(self, sample_path, eval_loader: DataLoader,):
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
                results["preds"].extend(list(logits.cpu()))
        
        auc = evaluation.evaluate(results["preds"], sample_path)
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
        print(f"evaluation loss: {average_loss:.5f}, auc: {auc * 100:2.2f}")
        self.model.train()
        return auc


# def auc_train(preds, labels):
#     model_outs = []
#     for i in range(len(preds)):
#         model_outs.append(preds[i].cpu().numpy()) # A 50D vector that assigns probability to each class

#     labels = np.array(labels.cpu()).astype(float)
#     model_outs = np.array(model_outs)
#     print(f"labels: {labels}")
#     print(f"models_out: {model_outs}")
#     auc_score = roc_auc_score(y_true=labels, y_score=model_outs)

#     return auc_score