from pytorch_lightning import Trainer
from LitAutoEncoder import LitAutoEncoder, Encoder, Decoder
from LitMNIST import LitMNISTDataset

dataset = LitMNISTDataset("./data")

model = LitAutoEncoder(Encoder(), Decoder())
trainer = Trainer(accelerator='gpu', devices=1, max_epochs=10)
trainer.fit(model=model, datamodule=dataset)
trainer.test(model=model, datamodule=dataset)