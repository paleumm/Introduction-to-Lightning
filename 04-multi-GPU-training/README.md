# Training our LightningModule with multiple GPUs

In this section, we are going to train our `LitAutoEncoder` on multiple GPUs. All the code can be copied directly from the last section, but there are some parts that have changed. And we are going to split our code into 3 files; `LitAutoEncoder.py`, `LitMNIST.py`, and `main.py`. 

In `LitMNIST.py`, nothing has changed.
```python
from pytorch_lightning import LightningDataModule
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class LitMNISTDataset(LightningDataModule):
    def __init__(self, root: str = "./data", batch_size: int = 32, num_workers=4):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers\
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    # Only on one GPU
    def prepare_data(self) -> None:
        datasets.MNIST(self.root, train=True, download=True) # Download the training set
        datasets.MNIST(self.root, train=False, download=True) # Download the test set

    # All GPU
    def setup(self, stage=None):
        self.mnist_test = datasets.MNIST(self.root, train=False, transform=self.transform)
        self.mnist_predict = datasets.MNIST(self.root, train=False, transform=self.transform)
        mnist_full = datasets.MNIST(self.root, train=True, download=True, transform=self.transform)
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size, num_workers=self.num_workers)
```

We'll add `sync_dist=True` inside the `self.log()` in LitAutoEncoder.py's **test_step** and **validation_step**. This is for syncing all the GPUs after each test or validation step.
```python
import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.l1(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        return self.l1(x)

from pytorch_lightning import LightningModule
from torch.optim import Adam

class LitAutoEncoder(LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("Traing-Loss : ", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("Test-Loss : ", loss, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("Val-Loss : ", loss, sync_dist=True)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer
```

The `__main__` entry point was added in `main.py`. The `device=-1` means we want all the GPUs we have.
```python
from pytorch_lightning import Trainer
from LitAutoEncoder import LitAutoEncoder, Encoder, Decoder
from LitMNIST import LitMNISTDataset

if __name__ == '__main__':
    dataset = LitMNISTDataset("./data")
    model = LitAutoEncoder(Encoder(), Decoder())
    trainer = Trainer(accelerator='gpu', devices=-1, max_epochs=10)
    trainer.fit(model=model, datamodule=dataset)
    trainer.test(model=model, datamodule=dataset)
```
