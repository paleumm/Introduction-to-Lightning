# LightningDataModule

In a typical Torch program, we can use the built-in datasets from **Torch**. But sometimes we need a custom dataset, so we have to write our own dataset class. **Lightning** provide us `LightningDataModule` that we can use it as our dataset with DataLoader directly with our `LightningModule`.

This is an example of how `LightningDataModule` can make our code easier to read.

```python
model = LitNeuralNetwork()
dataset = LitDataset(PATH_TO_DATASET)

trainer = Trainer()
trainer.fit(model=model, datamodule=dataset)
```

## Define DataModule

Importing
```python
from pytorch_lightning import LightningDataModule
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from torch import randperm
import torch.cuda as cuda

PATH = './data'
```

`__init__()` Define required parameters here. 

`root` : dataset directory

`batch_size` : batch size to use in DataLoader

```python
class LitMNISTDataset(LightningDataModule):
    def __init__(self, root: str = "./data", batch_size: int = 32):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        
```

`prepare_data()` define steps that should be done on only one GPU, like dowloading data.
`setup()` : define steps that should be done on every GPU, like splitting, transforming.

```python
class LitMNISTDataset(LightningDataModule):
    def __init__(self, root: str = "./data", batch_size: int = 32):
        ...
    
    # Only on one GPU
    def prepare_data(self) -> None:
        datasets.MNIST(self.root, train=True, download=True) # Download the training set
        datasets.MNIST(self.root, train=False, download=True) # Download the test set

    # All GPU
    def setup(self, stage=None):
        self.mnist_test = datasets.MNIST(self.root, train=False, transform=self.transform)
        self.mnist_predict = datasets.MNIST(self.root, train=False, transform=self.transform)
        mnist_full = datasets.MNIST(self.root, train=True, download=True, transform=self.transform)
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000]) # Split train and validation set.
```

#### DataLoader part

In **LightningDataModule** provided us the following type of **DataLoader**.

`train_dataloader` : for training set
`test_dataloader` : for test set
`val_dataloaer` : for validation set
`predict_dataloader` : for predicting

```python
class LitMNISTDataset(LightningDataModule):
    def __init__(self, root: str = "./data", batch_size: int = 32):
        ...
    
    def prepare_data(self) -> None:
        ...

    def setup(self, stage=None):
        ...

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)
    
    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)    
```

## Training

Let's use the **NeuralNetwork** model from the previous section. We recommend renaming it to `LitNeuralNetwork`. Why, I think it's cool and can identify that `Lit`  is **LightningDataModule**. The `validation_step` has been added in this tutorial.

```python
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim import SGD

class LitNeuralNetwork(LightningModule):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)     
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.linear_relu_stack(self.flatten(x))

    def training_step(self, batch, batch_idx):
        data, label = batch

        pred = self.linear_relu_stack(self.flatten(data))
        loss = self.loss_fn(pred, label)

        self.log("train-loss : ", loss)
        return loss

    def test_step(self, batch, batch_idx):
        data, label = batch

        pred = self.linear_relu_stack(self.flatten(data))
        loss = self.loss_fn(pred, label)

        self.log("test-loss : ", loss)

    # Don't forget to add this!!
    def validation_step(self, batch, batch_idx):
        data, label = batch

        pred = self.linear_relu_stack(self.flatten(data))
        loss = self.loss_fn(pred, label)

        self.log("val-loss : ", loss)

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=1e-3)
        return optimizer
```

This part will show us how LightningDataModule is used. 
```python
from pytorch_lightning import Trainer

dataset = LitMNISTDataset("./data")
model = LitNeuralNetwork()
trainer = Trainer(accelerator='gpu', devices=1, max_epochs=10)
trainer.fit(model=model, datamodule=dataset)
trainer.test(model=model, datamodule=dataset)
```