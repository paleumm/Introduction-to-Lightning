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

