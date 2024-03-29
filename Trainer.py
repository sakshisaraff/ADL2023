import time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from CNN import CNN
from CNN_extension import CNN_extension
from SampleCNN import SampleCNN

import evaluation

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Trainer:
    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            train_unshuffled: DataLoader,
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
        self.train_unshuffled = train_unshuffled
        self.inter_eval_loader = inter_eval_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
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
        # prints the number of parameters within the architecture
        print(f'Parameter Count: {count_parameters(self.model)}')
        best_auc = 0    
        #runs through each epoch
        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()
            #goes through each batch and calculates the logits
            #generates loss and auc score every eval-frequency
            #saves loss and auc to logs and prints in output
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
            
            self.summary_writer.add_scalar("epoch", epoch, self.step)
            if ((epoch + 1) % val_frequency) == 0 or (epoch + 1) >= epochs:
                train_auc = self.evaluate(Path("annotations/train_labels.pkl"), self.train_unshuffled, epoch)
                auc = self.evaluate(sample_path, self.inter_eval_loader, epoch)
                #implementation of early stopping- saves the model to a folder called model if the auc at evalutation if greater than the previous best validation auc
                if auc >= best_auc:
                    best_auc = auc
                    print("Saving model.")
                    torch.save({
                        'name': self.model.name,
                        'kwargs': self.model.kwargs, 
                        'args': hyperparameter_choices,
                        'model': self.model.state_dict(),
                        'auc': auc,
                        'epoch': epoch
                    }, model_path)
                # steps the scheduler if a scheduler is added
                if self.scheduler != None:
                    self.scheduler.step(auc)
                    print(self.scheduler._last_lr)

    #print loss every 10 steps of each epoch
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

    #logs loss every 10 steps of each epoch
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

    #evaluate our model at regular intervals during training
    #can pass training, validation or test dataset and it is able to handle them
    #saves the auc and average loss into summary writer/logs to allow for tensorboard to access them
    #used for model evaluation and training curves
    def evaluate(self, sample_path, eval_loader: DataLoader, epoch: int):
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

        if "train" in str(sample_path):
            self.summary_writer.add_scalars(
                "area under the curve",
                {"train": auc},
                epoch
            )
            self.summary_writer.add_scalars(
                "average loss",
                {"train": average_loss},
                epoch
            )
            print(f"train evaluation loss: {average_loss:.5f}, auc: {auc * 100:2.2f}")
        elif "val" in str(sample_path):
            self.summary_writer.add_scalars(
                "area under the curve",
                {"val": auc},
                epoch
            )
            self.summary_writer.add_scalars(
                "average loss",
                {"val": average_loss},
                epoch
            )
            self.summary_writer.add_scalars(
                "loss",
                {"val": average_loss},
                self.step
            )
            print(f"validation evaluation loss: {average_loss:.5f}, auc: {auc * 100:2.2f}")
        elif "test" in str(sample_path):
            self.summary_writer.add_scalars(
                "area under the curve",
                {"test": auc},
                epoch
            )
            self.summary_writer.add_scalars(
                "average loss",
                {"test": average_loss},
                epoch
            )
            self.summary_writer.add_scalars(
                "loss",
                {"test": average_loss},
                self.step
            )
            print(f"test evaluation loss: {average_loss:.5f}, auc: {auc * 100:2.2f}")        
        self.model.train()
        return auc

    #calls the best model from the models folder and evaluates it against the test dataset
    #can handle all the different models- basic, extension1, deep
    def test_evaluate(self, sample_path, model_path: Path, eval_loader: DataLoader):
        results = {"preds": []}
        total_loss = 0
        best_state = torch.load(model_path)
        best_model = None
        if best_state["name"] == "CNN_basic":
            best_model = CNN(**best_state["kwargs"])
        elif best_state["name"] == "CNN_extension1":
            best_model = CNN_extension(**best_state["kwargs"])
        elif best_state["name"] == "Deep":
            best_model = SampleCNN(**best_state["kwargs"])
        best_model.load_state_dict(best_state["model"])
        best_model = best_model.to(self.device)
        best_model.eval()
        with torch.no_grad():
            for filename, batch, labels in eval_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = best_model(batch)
                criterion = nn.BCELoss()
                loss = criterion(logits, labels)
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
        print(f"test evaluation loss: {average_loss:.5f}, best auc: {auc * 100:2.2f}")
        best_model.train()
        return auc