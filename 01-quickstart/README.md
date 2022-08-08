# Quickstart

This section will give  you an example of how `Lightning` can ease your PyTorch programming experience.

`Lightning` is based on `PyTorch` so you can use PyTorch inside Lightning code.

## Neural Network

This will import all the required libraries
```python
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.optim import SGD
```

Define `NeuralNetwork` class.

```python
class NeuralNetwork(LightningModule):
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
```
The LightningModule has many convenience methods, but the core ones you need to know about are:

`__init__` : Define computations here

`forward` : Use for inference only (separate from training_step)

`training_step` : the complete training loop

`validation_step` : the complete validation loop

`test_step` : the complete test loop

`predict_step` : the complete prediction loop

`configure_optimizers` : define optimizers and LR schedulers

In our `__init__`, we define `flatten`, `linear_relu_stack` (our neural network), and `loss_fn`.

Next is our `forward`, return the output from our network.
```python
class NeuralNetwork(LightningModule):
    def __init__(self): 
        ...
    
    def forward(self, x):
        return self.linear_relu_stack(self.flatten(x))
    
```

`training_step` has a significant role in Lightning. It defines how we want our network to train in each step. This will return a loss for each step.
```python
class NeuralNetwork(LightningModule):
    def __init__(self): 
        ...
    
    def forward(self, x):
        ...

    def training_step(self, batch, batch_idx):
        data, label = batch

        pred = self.linear_relu_stack(self.flatten(data))
        loss = self.loss_fn(pred, label)

        self.log("train-loss : ", loss)
        return loss
```

`test_step` and `validation_step` are similar to `training_step`
```python
class NeuralNetwork(LightningModule):
    def __init__(self): 
        ...
    
    def forward(self, x):
        ...

    def training_step(self, batch, batch_idx):
        ...

    def test_step(self, batch, batch_idx):
        data, label = batch

        pred = self.linear_relu_stack(self.flatten(data))
        loss = self.loss_fn(pred, label)

        self.log("test-loss : ", loss)

    def validation_step(self, batch, batch_idx):
        data, label = batch

        pred = self.linear_relu_stack(self.flatten(data))
        loss = self.loss_fn(pred, label)

        self.log("val-loss : ", loss)
```

By default, the `predict_step` method runs the `forward` method. In order to customize this behaviour, simply override the `predict_step` method. In this tutorial we will use the default one.

To choose which optimizers and learning-rate schedulers to use in your optimization, use the `configure_optimizers` method.

```python
class NeuralNetwork(LightningModule):
    def __init__(self): 
        ...
    
    def forward(self, x):
        ...

    def training_step(self, batch, batch_idx):
        ...

    def test_step(self, batch, batch_idx):
        ...

    def validation_step(self, batch, batch_idx):
        ...

    def configure_optimizers(self):
        optimizer = SGD(model.parameters(), lr=1e-3)
        return optimizer
```

## Training

This tutorial uses PyTorch's `FashionMNIST` dataset.

```python
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
```

We can use PyTorch's DataLoader with Lightning seamlessly.
```python
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
```

The main reason we prefer Lightning is convenience. If we have all the models and dataloaders, training is quite easy now!

```python
model = NeuralNetwork()
trainer = Trainer(accelerator='gpu', devices=1, max_epochs=10)
trainer.fit(model=model, train_dataloaders=train_dataloader)
trainer.test(model=model, dataloaders=test_dataloader)
```

That's all. The result is similar to pure PyTorch code.