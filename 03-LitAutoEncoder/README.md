# LitAutoEncoder

This section will implement the [AutoEncoder](https://en.wikipedia.org/wiki/Autoencoder) in Lightning.

### Encoder-Decoder

Define how `Encoder` and `Decoder` will look like in **Torch** by using `nn.module`.

```python
import torch
from torch import nn
import torch.nn.functional as F


```