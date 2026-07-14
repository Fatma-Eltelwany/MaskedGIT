# MaskedGIT
MaskedGIT Implementation using pytorch
maskgit/
│
├── README.md                    ← explain the project, tasks, results
│
├── synthetic/                   ← the toy tasks (clean, well-understood)
│   ├── data.py                  ← ConstantDataset, CopyDataset
│   ├── model.py                 ← MaskGITTransformer
│   ├── train.py                 ← training loop
│   ├── generate.py              ← iterative decoding
│   └── utils.py                 ← get_mask, cosine_schedule, mask_by_confidence
│
├── mnist/                       ← MNIST experiments
│   ├── VQVAE.py                 ← fixed VQVAE with EMA + dead code restart
│   ├── mnist_vqvae_data.py      ← unconditional dataloader
│   ├── cmnist_vqvae_data.py     ← conditional dataloader (returns labels)
│   ├── model.py                 ← unconditional MaskGIT
│   ├── cmodel.py                ← conditional MaskGIT (+ label embedding)
│   ├── train.py                 ← unconditional training
│   ├── ctrain.py                ← conditional training
│   ├── generate_and_decode.py   ← unconditional generation + VQVAE decode
│   ├── cgenerate_decode.py      ← conditional generation with CFG
│   ├── inpaint.py               ← inpainting experiment
│   └── diversity_check.py       ← collapse diagnostic tool
│
└── results/                     ← saved figures (commit the good ones)
    ├── copy_task_decoding.png
    ├── mnist_conditional.png
    ├── mnist_cfg_weight.png
    └── codebook_usage.png
