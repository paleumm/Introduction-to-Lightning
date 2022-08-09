from DataModule import LitMNISTDataset
from Model import NeuralNetwork
from pytorch_lightning import Trainer

dataset = LitMNISTDataset("./data")
model = NeuralNetwork()
trainer = Trainer(accelerator='gpu', devices=1, max_epochs=10)
trainer.fit(model=model, datamodule=dataset)
trainer.test(model=model, datamodule=dataset)