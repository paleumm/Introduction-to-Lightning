from pytorch_lightning import LightningModule
from torch import nn
from torch.optim import SGD
from ResNet import ResNet, block
from torchmetrics import Accuracy

class LitResNet(LightningModule):
    def __init__(self, resnet_type:list = [3, 4, 6, 3], num_classes = 1000, image_channels = 3):
        super().__init__()
        self.ResNet = ResNet(block, resnet_type, image_channels, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()

    def forward(self, x):
        return self.ResNet(x)

    def training_step(self, batch, batch_idx):
        data, label = batch

        pred = self.ResNet(data)
        loss = self.loss_fn(pred, label)
        batch_value = self.train_acc(pred, label)
        self.log('train_acc', batch_value)
        return loss 
    
    def training_epoch_end(self, outputs):
        self.train_acc.reset()

    def test_step(self, batch, batch_idx):
        data, label = batch

        pred = self.ResNet(data)
        loss = self.loss_fn(pred, label)

        self.log("test-loss : ", loss)

    def validation_step(self, batch, batch_idx):
        data, label = batch

        pred = self.ResNet(data)
        loss = self.loss_fn(pred, label)

        self.valid_acc(pred, label)
        self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=True)

    def validation_epoch_end(self, outputs):
        self.log('valid_acc_epoch', self.valid_acc.compute())
        self.valid_acc.reset()

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=1e-3)
        return optimizer

class LitResNet50(LitResNet):
    def __init__(self, num_classes = 1000, image_channels = 3):
        super(LitResNet50, self).__init__(num_classes=num_classes, image_channels=image_channels,resnet_type=[3, 4, 6, 3])
        

class LitResNet101(LitResNet):
    def __init__(self, num_classes = 1000, image_channels = 3):
        super(LitResNet101, self).__init__(num_classes=num_classes, image_channels=image_channels,resnet_type=[3, 4, 23, 3])

class LitResNet152(LitResNet):
    def __init__(self, num_classes = 1000, image_channels = 3):
        super(LitResNet152, self).__init__(num_classes=num_classes, image_channels=image_channels,resnet_type=[3, 8, 36, 3])