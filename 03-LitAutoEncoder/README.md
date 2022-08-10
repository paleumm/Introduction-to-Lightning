# LitAutoEncoder

This section will implement the [AutoEncoder](https://en.wikipedia.org/wiki/Autoencoder) in Lightning.

### Encoder-Decoder

Define how `Encoder` and `Decoder` will look like in **Torch** by using `nn.module`.

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
```

Then we can define our `LitAutoEncoder`

```python
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
        self.log("Test-Loss : ", loss)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("Val-Loss : ", loss)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer
```

We use our last tutorial's DataModule, and rename it to `LitMNISTDataset`.

```python
from pytorch_lightning import LightningDataModule
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class LitMNISTDataset(LightningDataModule):
    ...
```

And all the training stuff are similar to the last tutorial.
```python
from pytorch_lightning import Trainer

dataset = LitMNISTDataset("./data")

model = LitAutoEncoder(Encoder(), Decoder())
trainer = Trainer(accelerator='gpu', devices=1, max_epochs=10)
trainer.fit(model=model, datamodule=dataset)
trainer.test(model=model, datamodule=dataset)
```