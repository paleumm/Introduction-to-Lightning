from pytorch_lightning import LightningDataModule
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch import randperm

VALID_SPLIT = 0.1
IMAGE_SIZE = 224

PATH = "./data"
TRAIN_TRANSFORM = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

VALID_TRANSFORM = valid_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

TRANSFORM = [TRAIN_TRANSFORMS, VALID_TRANSFORMS]

BATCH_SIZE = 64
NUM_WORKER = 16

class LitImageNetDataset(LightningDataModule):
    def __init__(self, path:str = PATH, transform = None, train_transform=TRAIN_TRANSFORM, valid_transform=VALID_TRANSFORM, num_classes=None, batch_size=BATCH_SIZE, num_workers=NUM_WORKER):
        super().__init__()
        self.path = path
        self.transform = transform
        self.train_transform=train_transform
        self.valid_transform=valid_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = num_classes
        if self.num_classes is None:
            self.num_classes = len(next(os.walk('dir_name'))[1])
        
    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage=None):
        dataset = datasets.ImageFolder(
            self.data_path, 
            transform=self.train_transform
        )
        dataset_val = datasets.ImageFolder(
            self.data_path, 
            transform=self.valid_transform
        )
        dataset_test = datasets.ImageFolder(
            self.test_path,
            transform=self.valid_transform
        )
        dataset_size = len(dataset)
        valid_size = int(VALID_SPLIT*dataset_size)

        indices = randperm(len(dataset)).tolist()
        self.dataset_train = Subset(dataset, indices[:-valid_size])
        self.dataset_valid = Subset(dataset_val, indices[-valid_size:])
        self.dataset_test = dataset_test

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset_valid, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        pass