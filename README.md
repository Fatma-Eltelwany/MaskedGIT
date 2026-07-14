# MaskedGIT

A PyTorch implementation of **MaskedGIT (Masked Generative Image Transformer)** inspired by the CVPR 2022 paper. This repository explores the MaskedGIT architecture from first principles, beginning with simple synthetic tasks and progressing to image generation experiments on MNIST using a VQ-VAE tokenizer.

## Repository Structure

```text
maskgit/
├── README.md                    # Project overview
├── synthetic/                   # Constant & Copy tasks
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── generate.py
│   └── utils.py
├── mnist/                       # MNIST experiments
│   ├── VQVAE.py
│   ├── mnist_vqvae_data.py
│   ├── model.py
│   ├── train.py
│   ├── generate_and_decode.py
│   └── mnist_trial_results.ipynb
└── results/                     # Generated figures
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
