from pytorch_lightning import LightningDataModule
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class LitMNISTDataset(LightningDataModule):
    def __init__(self, root: str = "./data", batch_size: int = 32, num_workers=4):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers

        # transform PIL to Tensor, so it can be use with the model
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
    