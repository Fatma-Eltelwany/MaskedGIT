import torch
from torch.utils.data import Dataset, DataLoader


class MNISTVQVAETokenDataset(Dataset):
    def __init__(self, path: str = "mnist_vqvae_tokens_train.pt"):
        data         = torch.load(path)
        self.tokens  = data["tokens"]   # (N, 49)
        self.labels  = data["labels"]   # (N,) digits 0-9

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx], self.labels[idx]   # ← returns tuple now


def get_dataloader(batch_size=128, shuffle=True,
                   path="/u/fatel/HEP4M/MaskedGIT/mnist/mnist_vqvae_tokens_train.pt"):
    dataset = MNISTVQVAETokenDataset(path)
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=shuffle, num_workers=4, pin_memory=True)