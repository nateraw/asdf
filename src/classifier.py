import pytorch_lightning as pl
import torch
from torch import nn


class Classifier(pl.LightningModule):

    def __init__(self, model=None, learning_rate=0.001, criterion=None, train_metric=None, val_metric=None, test_metric=None):
        super().__init__()
        self.save_hyperparameters()
        self.model = model or nn.Linear(784, 10)
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.train_metric = train_metric or pl.metrics.Accuracy()
        self.val_metric = val_metric or pl.metrics.Accuracy()
        self.test_metric = test_metric or pl.metrics.Accuracy()
        self.metrics = dict(train=self.train_metric, val=self.val_metric, test=self.test_metric)

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, split="train"):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)
        self.log(f"{split}_loss", loss)

        if split in ["val", "test"]:
            preds = outputs.argmax(dim=1)
            acc = self.metrics[split](preds, y)
            self.log(f"{split}_acc", acc, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    
    def fit(self, *args, **kwargs):
        pl.Trainer(**kwargs).fit(self, *args)
