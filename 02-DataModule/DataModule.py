from pytorch_lightning import LightningDataModule
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class LitMNISTDataset(LightningDataModule):
    def __init__(self, root: str = "./data", batch_size: int = 32):
        super().__init__()
        self.root = root
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage=None):
        self.mnist_test = datasets.MNIST(self.root, train=False, download=True, transform=transforms.ToTensor())
        self.mnist_predict = datasets.MNIST(self.root, train=False, download=True, transform=transforms.ToTensor())
        mnist_full = datasets.MNIST(self.root, train=True, download=True, transform=transforms.ToTensor())
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    # def val_dataloader(self):
    #     return DataLoader(self.mnist_val, batch_size=self.batch_size)
    
    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)
    