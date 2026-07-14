# MaskedGIT

A PyTorch implementation of **MaskedGIT (Masked Generative Image Transformer)** inspired by the CVPR 2022 paper. This repository explores the MaskedGIT architecture from first principles, beginning with simple synthetic tasks and progressing to image generation experiments on MNIST using a VQ-VAE tokenizer.

## Repository Structure

```text
maskgit/
│
├── README.md                    # Project overview, methodology, and results
│
├── Synthetic Tasks (Constant & Copy Tasks)/                   # Synthetic toy tasks for validating the algorithm
│   ├── data.py                  # ConstantDataset, CopyDataset
│   ├── model.py                 # MaskGIT Transformer
│   ├── train.py                 # Training loop
│   ├── generate.py              # Iterative parallel decoding
│   └── utils.py                 # Mask scheduling and confidence-based remasking
│
├── mnist/                       # MNIST experiments
│   ├── VQVAE.py                 # VQ-VAE with EMA updates and dead-code replacement
│   ├── mnist_vqvae_data.py      # Data loader
│   ├── model.py                 # MaskGIT model
│   ├── train.py                 # Training
│   ├── generate_and_decode.py   # Image generation and VQ-VAE decoding
│   └── mnist_trial_results.ipynb # Notebook to show results


```

## Features

- PyTorch implementation of MaskedGIT
- Iterative parallel decoding
- Confidence-based token remasking
- Cosine masking schedule
- Synthetic toy tasks for algorithm verification
- Unconditional and conditional image generation on MNIST
- Classifier-Free Guidance (CFG)
- Image inpainting
- VQ-VAE tokenizer with EMA codebook updates
- Codebook usage and diversity analysis
