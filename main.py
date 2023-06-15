import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from argparse import ArgumentParser
from pytorch_lightning import loggers

# Define Dataset
class FaceDataset(Dataset):
    def __init__(self, X, y):
        self.sample = torch.from_numpy(X).float()
        self.target = torch.from_numpy(y).float()

    def __getitem__(self, idx):
        return self.sample[idx], self.target[idx]

    def __len__(self):
        return len(self.sample)

# Define DataModule
class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    # If needs to load different data, modify here
    def setup(self, stage=None):
        # Load data
        X = np.load("norm_mel.npy")
        y = np.load("norm_landmark.npy")
        dataset = FaceDataset(X, y)
        # Split training data and validation data
        self.train_dataset, self.valid_dataset, self.test_dataset = random_split(dataset, [
                                                              0.8, 0.1, 0.1])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
    
# Define network
class DNN(pl.LightningModule):
    def __init__(self, lr, n_in, n_hidden, n_out, d_rate=0):
        super(DNN, self).__init__()
        self._train_outputs = []
        self._valid_outputs = []

        self.save_hyperparameters()
        self.learning_rate = lr
        self.dropout_rate = d_rate
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_out)
        self.dropout = nn.Dropout(self.dropout_rate)
        
    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        y_predict = self(X)
        loss = F.mse_loss(y_predict,y)
        self._train_outputs.append(loss)
        return loss
    
    def on_train_epoch_end(self):
        avg_loss = torch.stack(self._train_outputs).mean()
        self.log_dict({"train_loss":avg_loss, "step":float(self.current_epoch+1)}, prog_bar=True)
        self._train_outputs.clear()
        
    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_predict = self(X)
        loss = F.mse_loss(y_predict,y)
        self._valid_outputs.append(loss)
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self._valid_outputs).mean()
        self.log_dict({"valid_loss":avg_loss, "step":float(self.current_epoch+1)}, prog_bar=True)
        self._valid_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        X, y = batch
        y_predict = self(X)
        loss = F.mse_loss(y_predict,y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

def main(args):
    # Hyperparameters
    # fix hp
    n_in = 40
    n_out = 80
    learning_rate = 1e-3
    # tune hp
    n_hidden = 512
    batch_size = 64
    num_epochs = 100
    
    dataset = DataModule(batch_size=args.batch_size)
    net = DNN(lr=args.lr, n_in=n_in, n_hidden=args.n_hidden, n_out=n_out, d_rate=args.d_rate)
    callbacks = [ModelCheckpoint(save_top_k=1, mode='min', monitor="valid_loss")]  # save top 1 model
    tb_logger = loggers.TensorBoardLogger(save_dir="./", name=f"expt{args.expt}")
    trainer = pl.Trainer(max_epochs=args.epochs, callbacks=callbacks, logger=tb_logger)
    trainer.fit(model=net, datamodule=dataset)
    
    # show the min validation loss
    best_validation_loss = callbacks[0].best_model_score
    print(f"\n\nMin validation loss: {best_validation_loss.item()}\n\n")
    trainer.test(model=net, datamodule=dataset, ckpt_path='best')

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--expt", default=1, type=int)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--n_hidden", default=128, type=int)
    parser.add_argument("--d_rate", default=0, type=float)
    parser.add_argument("--lr", default=1e-3, type=float)
    
    
    args = parser.parse_args()

    main(args)


    # python main.py --expt 5 --epochs 500 --batch_size 64 --n_hidden 512 --d_rate 0.05 --lr 1e-3
    