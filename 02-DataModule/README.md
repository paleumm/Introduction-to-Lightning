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

In `__init__` we
```python
class LitMNISTDataset(LightningDataModule):
    def __init__(self, root: str = PATH, batch_size: int = 32):
        self.root = root
        self.batch_size = batch_size
        
```