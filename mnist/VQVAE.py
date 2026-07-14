# VQVAE.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([transforms.ToTensor()])


class VectorQuantizerEMA(nn.Module):
    """
    Vector Quantizer with Exponential Moving Average (EMA) codebook updates
    and dead code restart.

    Key differences from original:
        1. Codebook updated via EMA, not gradients
           → more stable, avoids codebook collapse
        2. Dead codes (unused for > threshold batches) get reset
           → breaks the rich-get-richer feedback loop
        3. Encoder output L2-normalized before quantization
           → all codes compete on equal footing

    Paper reference:
        "Neural Discrete Representation Learning" (van den Oord et al. 2017)
        EMA update variant from Appendix A.1
    """

    def __init__(self, num_embeddings, embedding_dim, beta=0.25,
                 decay=0.99, epsilon=1e-5, dead_code_threshold=1.0):
        super().__init__()
        self.num_embeddings      = num_embeddings
        self.embedding_dim       = embedding_dim
        self.beta                = beta
        self.decay               = decay
        self.epsilon             = epsilon
        self.dead_code_threshold = dead_code_threshold

        # Codebook — not a parameter, updated via EMA
        embed = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("embedding",    embed)
        self.register_buffer("cluster_size", torch.ones(num_embeddings))
        self.register_buffer("embed_avg",    embed.clone())

    def forward(self, z):
        # z: (B, C, H, W)
        B, C, H, W = z.shape

        # Normalize encoder output — helps all codes compete fairly
        z_normalized = F.normalize(z, dim=1)

        # Flatten to (B*H*W, C)
        z_flat = z_normalized.permute(0, 2, 3, 1).contiguous().view(-1, C)

        # Distances to codebook
        dist = (
            z_flat.pow(2).sum(1, keepdim=True)
            + self.embedding.pow(2).sum(1)
            - 2 * z_flat @ self.embedding.t()
        )

        # Nearest code
        encoding_indices = dist.argmin(dim=1)             # (B*H*W,)
        encodings = F.one_hot(
            encoding_indices, self.num_embeddings
        ).float()                                          # (B*H*W, num_embeddings)

        # Quantized output
        z_q = (encodings @ self.embedding).view(B, H, W, C)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()       # (B, C, H, W)

        # ── EMA codebook update (training only) ───────────────────────────
        if self.training:
            # Update cluster sizes
            cluster_size = encodings.sum(0)               # (num_embeddings,)
            self.cluster_size = (
                self.decay * self.cluster_size
                + (1 - self.decay) * cluster_size
            )

            # Update embedding averages
            embed_sum = encodings.t() @ z_flat            # (num_embeddings, C)
            self.embed_avg = (
                self.decay * self.embed_avg
                + (1 - self.decay) * embed_sum
            )

            # Laplace smoothing for stability
            n = self.cluster_size.sum()
            smoothed = (
                (self.cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon) * n
            )
            self.embedding = self.embed_avg / smoothed.unsqueeze(1)

            # ── Dead code restart ─────────────────────────────────────────
            # Any code with cluster_size < threshold gets reset to a
            # random encoder output from the current batch
            dead_codes = self.cluster_size < self.dead_code_threshold
            n_dead     = dead_codes.sum().item()
            if n_dead > 0:
                # Sample random encoder outputs to replace dead codes
                random_idx      = torch.randint(
                    len(z_flat), (n_dead,), device=z.device
                )
                random_encodings = z_flat[random_idx]
                self.embedding[dead_codes]  = random_encodings
                self.embed_avg[dead_codes]  = random_encodings
                self.cluster_size[dead_codes] = self.dead_code_threshold

        # Commitment loss only (codebook updated via EMA, not gradients)
        loss = self.beta * F.mse_loss(z_q.detach(), z_normalized)

        # Straight-through estimator
        z_q_st = z_normalized + (z_q - z_normalized).detach()

        return z_q_st, loss, encoding_indices


class Encoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 4, 2, 1),   # 28→14
            nn.GroupNorm(8, hidden_dim),                    # more stable than BatchNorm
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 4, 2, 1),   # 14→7
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),   # 7→7, refine
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1),          # pointwise projection
        )

    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, out_channels=1, hidden_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, 2, 1),   # 7→14
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, 2, 1),   # 14→28
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, out_channels, 1),
        )

    def forward(self, z):
        return self.conv(z)


class VQVAE(nn.Module):
    def __init__(self, num_embeddings=128, embedding_dim=64, in_channels=1):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels, hidden_dim=embedding_dim)
        self.vq      = VectorQuantizerEMA(num_embeddings, embedding_dim)
        self.decoder = Decoder(out_channels=in_channels, hidden_dim=embedding_dim)

    def forward(self, x):
        z     = self.encoder(x)
        z_q, vq_loss, indices = self.vq(z)
        x_hat = self.decoder(z_q)
        return x_hat, vq_loss, indices


def encode_to_tokens(model, x):
    with torch.no_grad():
        z = model.encoder(x)
        # normalize before quantization, matching VectorQuantizerEMA
        z_normalized = F.normalize(z, dim=1)
        _, _, indices = model.vq(z_normalized.unsqueeze(0).squeeze(0))
    return indices.view(x.shape[0], -1)


def tensor_to_image(x):
    x = x.detach().cpu().permute(1, 2, 0).clamp(0, 1)
    if x.shape[-1] == 1:
        x = x.squeeze(-1)
    return x


def show_reconstructions(x, x_hat, n=8):
    fig, axes = plt.subplots(2, n, figsize=(2*n, 4))
    for i in range(n):
        axes[0, i].imshow(tensor_to_image(x[i]),     cmap="gray", vmin=0, vmax=1)
        axes[1, i].imshow(tensor_to_image(x_hat[i]), cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original",      fontsize=10)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=10)
    plt.suptitle("VQVAE Reconstruction Quality", fontsize=12)
    plt.tight_layout()
    plt.savefig("vqvae_reconstruction.png", dpi=150)
    plt.show()


def encode_full_dataset(model, dataset, device, batch_size=256):
    loader     = DataLoader(dataset, batch_size=batch_size,
                           shuffle=False, num_workers=4)
    all_tokens = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x      = x.to(device)
            z      = model.encoder(x)
            z_norm = F.normalize(z, dim=1)
            _, _, indices = model.vq(z_norm)
            tokens = indices.view(x.shape[0], -1).cpu()
            all_tokens.append(tokens)
            all_labels.append(y)
    return torch.cat(all_tokens), torch.cat(all_labels)


def check_codebook_usage(tokens, num_embeddings=128, split="train"):
    flat   = tokens.flatten().tolist()
    from collections import Counter
    counts = Counter(flat)
    usage  = len(counts)
    top1   = max(counts.values()) / len(flat) * 100

    print(f"\nCodebook usage ({split}):")
    print(f"  Unique codes used : {usage}/{num_embeddings}")
    print(f"  Top-1 frequency   : {top1:.1f}%  (uniform = {100/num_embeddings:.1f}%)")
    print(f"  Top-5 codes       : {counts.most_common(5)}")

    # Plot
    all_counts = [counts.get(c, 0) / len(flat) * 100 for c in range(num_embeddings)]
    plt.figure(figsize=(12, 3))
    colors = ["#C00000" if counts.get(c, 0) == 0 else "#1E3A5F"
              for c in range(num_embeddings)]
    plt.bar(range(num_embeddings), all_counts, color=colors, width=0.9)
    plt.axhline(100/num_embeddings, color="green", linestyle="--",
                label=f"Uniform ({100/num_embeddings:.1f}%)")
    plt.xlabel("Codebook index")
    plt.ylabel("Frequency (%)")
    plt.title(f"VQVAE Codebook Usage ({split}) — "
              f"{usage}/{num_embeddings} codes active")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"codebook_usage_{split}.png", dpi=150)
    plt.show()
    return usage, top1


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_ds = datasets.MNIST(root="./data", train=True,
                              download=True, transform=transform)
    loader   = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)

    model     = VQVAE(num_embeddings=128, embedding_dim=64, in_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    # Check latent shape
    with torch.no_grad():
        z = model.encoder(next(iter(loader))[0][:1].to(device))
    print(f"Latent grid: {z.shape} → seq_len = {z.shape[2]*z.shape[3]}")

    # ── Training loop ─────────────────────────────────────────────────────
    n_epochs = 30   # more epochs than before
    for epoch in range(n_epochs):
        model.train()
        total_recon = total_vq = 0.0
        n_dead_total = 0

        for x, _ in loader:
            x = x.to(device)

            x_hat, vq_loss, _ = model(x)
            recon_loss = F.mse_loss(x_hat, x)
            loss       = recon_loss + vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_recon  += recon_loss.item()
            total_vq     += vq_loss.item()

        # Check dead codes each epoch
        dead = (model.vq.cluster_size < model.vq.dead_code_threshold).sum().item()
        print(f"Epoch {epoch+1:>3}/{n_epochs}  "
              f"recon: {total_recon/len(loader):.4f}  "
              f"vq: {total_vq/len(loader):.4f}  "
              f"dead codes: {dead}/128")

    # ── Save model ────────────────────────────────────────────────────────
    torch.save(model.state_dict(), "vqvae_mnist.pt")
    print("\nSaved vqvae_mnist.pt")

    # ── Visual check ─────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        x, _ = next(iter(loader))
        x     = x.to(device)
        x_hat, _, _ = model(x)
    show_reconstructions(x.cpu(), x_hat.cpu(), n=8)

    # ── Encode and check codebook usage ──────────────────────────────────
    print("\nPre-encoding train set...")
    train_tokens, train_labels = encode_full_dataset(model, train_ds, device)
    usage, top1 = check_codebook_usage(train_tokens, split="train")

    torch.save({"tokens": train_tokens, "labels": train_labels},
               "mnist_vqvae_tokens_train.pt")

    test_ds = datasets.MNIST(root="./data", train=False,
                             download=True, transform=transform)
    test_tokens, test_labels = encode_full_dataset(model, test_ds, device)
    torch.save({"tokens": test_tokens, "labels": test_labels},
               "mnist_vqvae_tokens_test.pt")

    check_codebook_usage(test_tokens, split="test")

    print(f"\nFinal codebook health:")
    print(f"  Unique codes: {usage}/128")
    print(f"  Top-1 freq  : {top1:.1f}%  (target: <5%)")
    print(f"  {'✓ Codebook balanced!' if top1 < 10 else '✗ Still imbalanced — consider increasing n_epochs or embedding_dim'}")