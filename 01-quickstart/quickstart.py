from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.optim import SGD

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

    def configure_optimizers(self):
        optimizer = SGD(model.parameters(), lr=1e-3)
        return optimizer

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
BATCH_SIZE = 64
NUM_WORKER = 4
# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKER)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKER)

model = NeuralNetwork()
trainer = Trainer(accelerator='gpu', devices=1, max_epochs=10)
trainer.fit(model=model, train_dataloaders=train_dataloader)
trainer.test(model=model, dataloaders=test_dataloader)
