from pytorch_lightning import LightningModule
from torch import nn
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
        optimizer = SGD(self.parameters(), lr=1e-3)
        return optimizer