import torch
from tqdm import tqdm
from data import get_dataloaders
import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

class Trainer:
    def __init__(
        self, 
        model: torch.nn.Module, 
        train_loader: torch.utils.data.DataLoader, 
        test_loader: torch.utils.data.DataLoader, 
        optimizer: str, 
        criterion: str, 
        num_epochs: int,
        results_dir: str):

        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs

        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        elif optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        else:
            raise ValueError(f"Optimizer {optimizer} not supported, must be one of: adam, sgd")

        if criterion == "cross_entropy":
            self.criterion = torch.nn.CrossEntropyLoss()
        elif criterion == "mse":
            self.criterion = torch.nn.MSELoss()
        else:
            raise ValueError(f"Criterion {criterion} not supported, must be one of: cross_entropy, mse")

        # create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)

        self.results_dir = results_dir
        self.log_file = os.path.join(results_dir, "logs.csv")

        # initialize log file
        with open(self.log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "epoch", "train_loss", "test_loss"])


    def train(self):

        epoch_pb = tqdm(range(self.num_epochs), desc="Epochs")

        for epoch in epoch_pb:

            train_pb = tqdm(self.train_loader, desc="Training", leave=False)

            running_train_loss = 0.0
            running_test_loss = 0.0

            self.model.train()

            for batch_idx, batch in enumerate(train_pb):

                images, labels = batch
                self.optimizer.zero_grad()
                outputs = self.model(images)
                train_loss = self.criterion(outputs, labels)
                train_loss.backward()
                
                self.optimizer.step()

                running_train_loss += train_loss.item()

            test_pb = tqdm(self.test_loader, desc="Testing", leave=False)

            for batch_idx, batch in enumerate(test_pb):

                self.model.eval()

                images, labels = batch
                outputs = self.model(images)
                test_loss = self.criterion(outputs, labels)

                running_test_loss += test_loss.item()

            epoch_pb.set_postfix(train_loss=running_train_loss / len(self.train_loader), test_loss=running_test_loss / len(self.test_loader))

            self.add_log(epoch, running_train_loss / len(self.train_loader), running_test_loss / len(self.test_loader))

    def add_log(self, epoch: int, train_loss: float, test_loss: float):

        with open(self.log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([datetime.now(), epoch, train_loss, test_loss])

    def plot_losses(self):

        # read logs
        logs = pd.read_csv(self.log_file)

        # plot losses
        plt.figure(figsize=(10, 5))
        plt.plot(logs['epoch'], logs['train_loss'], label='Train Loss')
        plt.plot(logs['epoch'], logs['test_loss'], label='Test Loss')
        plt.legend()
        plt.savefig(os.path.join(self.results_dir, "losses.png"))


if __name__ == "__main__":

    train_loader, test_loader = get_dataloaders(
        train_images_path="~/data/mnist/train-images.idx3-ubyte",
        train_labels_path="~/data/mnist/train-labels.idx1-ubyte",
    )
