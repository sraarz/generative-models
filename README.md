# Generative Models: Autoencoders and GANs

PyTorch implementations of autoencoder and adversarial architectures on CIFAR-10
and FashionMNIST.

## Autoencoders

### [`one-input-two-outputs-autoencoder.ipynb`](notebooks/one-input-two-outputs-autoencoder.ipynb)
A dual-head autoencoder: one shared encoder, two decoder branches reconstructing
two different targets from a single input image. Both heads are trained jointly by
summing their MSE losses before a single `backward()` pass, so the shared latent
representation must serve both reconstruction objectives at once.

### [`NN_fourth_assignment_CIFAR10_AE.ipynb`](notebooks/NN_fourth_assignment_CIFAR10_AE.ipynb)
A convolutional autoencoder on CIFAR-10 with explicit `train`/`test` loops, a
`combineImages` utility for composing input batches, and side-by-side visual
comparison of originals against reconstructions.

## GANs

| Notebook | Architecture |
|---|---|
| [`GAN.ipynb`](notebooks/GAN.ipynb) | Fully-connected generator/discriminator baseline |
| [`GAN_fashionMNIST.ipynb`](notebooks/GAN_fashionMNIST.ipynb) | MLP GAN trained on FashionMNIST |
| [`ConvGAN_fashionMNIST.ipynb`](notebooks/ConvGAN_fashionMNIST.ipynb) | Convolutional GAN (transposed-conv generator, strided-conv discriminator) |

Each follows the standard alternating training loop — discriminator on real and
detached-fake batches, then generator against the updated discriminator — with
generated-sample grids logged across epochs to show sample quality developing.
The convolutional variant produces visibly sharper samples than the MLP baseline.

## Running

```bash
pip install torch torchvision matplotlib numpy pandas
```

CIFAR-10 and FashionMNIST download automatically via `torchvision.datasets` on
first run. A GPU is recommended for the GAN notebooks.

## Author

Neural networks coursework. `ConvGAN_fashionMNIST.ipynb` and
`GAN_fashionMNIST.ipynb` are by Sara Rezaei.
