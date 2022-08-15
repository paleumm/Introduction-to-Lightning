from turtle import forward
from pytorch_lightning import LightningModule
from torch import nn, cuda, randn
from torch.optim import SGD

'''
PyTorch's Resnet implementation is reference from https://www.analyticsvidhya.com/blog/2021/06/build-resnet-from-scratch-with-python/
'''

resnet = {'ResNet50':[3, 4, 6, 3], 'ResNet101':[3, 4, 23, 3], 'ResNet152':[3, 8, 36, 3]}
default_resnet = 'ResNet50'

class LitResNet(LightningModule):
    def __init__(self, resnet_type:str = default_resnet, num_classes = 1000, image_channels = 3):
        super().__init__()
        self.ResNet = ResNet(block, resnet[default_resnet], image_channels, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.ResNet(x)

    def training_step(self, batch, batch_idx):
        data, label = batch

        pred = self.ResNet(data)
        loss = self.loss_fn(pred, label)

        self.log("train-loss : ", loss)
        return loss 

    def test_step(self, batch, batch_idx):
        data, label = batch

        pred = self.ResNet(data)
        loss = self.loss_fn(pred, label)

        self.log("test-loss : ", loss)

    def validation_step(self, batch, batch_idx):
        data, label = batch

        pred = self.ResNet(data)
        loss = self.loss_fn(pred, label)

        self.log("val-loss : ", loss)

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=1e-3)
        return optimizer

class block(nn.Module):
    def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=1) -> None:
        super(block, self).__init__()

        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=intermediate_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            in_channels=intermediate_channels, 
            out_channels=intermediate_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1, 
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            in_channels=intermediate_channels, 
            out_channels=intermediate_channels * self.expansion, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.layers = nn.Sequential(
            self.conv1, self.bn1, self.conv2, self.bn2, self.conv3, self.bn3,
        )
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.layers(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes) -> None:
        super(ResNet, self).__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=image_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

        self.layers = nn.Sequential(
            self.conv1, self.bn1, self.relu, self.maxpool, 
            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool
        )


    def forward(self, x):
        x = self.layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=intermediate_channels * 4, kernel_size=1, stride=stride, bias=False), 
                nn.BatchNorm2d(intermediate_channels * 4)
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride, )
        )
        self.in_channels = intermediate_channels * 4

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))
        
        return nn.Sequential(*layers)
    
def ResNet50(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)

def ResNet101(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 23, 3], img_channel, num_classes)

def ResNet152(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 8, 36, 3], img_channel, num_classes)

# def test():
#     net = ResNet101(img_channel=3, num_classes=1000)
#     device = "cuda" if cuda.is_available() else "cpu"
#     y = net(randn(4, 3, 224, 224)).to(device)
#     print(y.size())

# test()