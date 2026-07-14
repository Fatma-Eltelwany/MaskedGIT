import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from cmodel import MaskGITTransformer
from utils import cosine_schedule, mask_by_confidence
from VQVAE import VQVAE

BASE_DIR   = os.environ.get("BASE_DIR",   "/viper/u2/fatel/HEP4M/MaskedGIT/mnist")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(BASE_DIR, "checkpoints"))

VOCAB_SIZE = 128
SEQ_LEN    = 49
GRID_SIZE  = 7
MASK_ID    = VOCAB_SIZE


@torch.no_grad()
def generate_tokens(model, labels, T=20, temperature=1.0,
                    cfg_weight=3.0):
    """
    Class-conditional generation with classifier-free guidance.

    CFG formula (from reference code):
        logit = (1 + w) * logit_conditional - w * logit_unconditional
    """
    batch_size  = len(labels)
    device      = next(model.parameters()).device
    labels      = torch.tensor(labels, dtype=torch.long, device=device)
    tokens      = torch.full((batch_size, SEQ_LEN), MASK_ID,
                             dtype=torch.long, device=device)
    null_labels = torch.full_like(labels, model.NULL_CLASS)
    no_drop     = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for t in range(1, T + 1):
        if cfg_weight > 0:
            # Run model twice — once with label, once without
            logit_c = model(tokens, labels,      no_drop)[:, :, :MASK_ID]
            logit_u = model(tokens, null_labels, no_drop)[:, :, :MASK_ID]
            logits  = (1 + cfg_weight) * logit_c - cfg_weight * logit_u
        else:
            logits = model(tokens, labels, no_drop)[:, :, :MASK_ID]

        logits     = logits / temperature
        probs      = F.softmax(logits, dim=-1)
        sampled    = torch.multinomial(
            probs.view(-1, MASK_ID), 1
        ).view(batch_size, SEQ_LEN)
        confidence = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

        masked     = tokens == MASK_ID
        tokens     = torch.where(masked, sampled, tokens)
        confidence = torch.where(masked, confidence, torch.ones_like(confidence))

        n_to_mask = cosine_schedule(t, T, SEQ_LEN)
        new_mask  = mask_by_confidence(confidence, n_to_mask)
        tokens[new_mask] = MASK_ID

    return tokens


def decode_tokens(vqvae, tokens):
    """Token sequences → pixel images via VQVAE decoder."""
    B = tokens.shape[0]
    with torch.no_grad():
        codes  = tokens.view(B, GRID_SIZE, GRID_SIZE)
        z_q    = vqvae.vq.embedding[codes]              # (B, 7, 7, emb_dim)
        z_q    = z_q.permute(0, 3, 1, 2).contiguous()  # (B, emb_dim, 7, 7)
        images = vqvae.decoder(z_q).clamp(0, 1)        # (B, 1, 28, 28)
    return images


# ── Load MaskGIT ──────────────────────────────────────────────────────────────
print("Loading MaskGIT...")
ckpt = torch.load(os.path.join(OUTPUT_DIR, "maskgit_cond.pt"), map_location="cpu")
maskgit = MaskGITTransformer(
    vocab_size = ckpt["vocab_size"],
    seq_len    = ckpt["seq_len"],
    n_classes  = ckpt["n_classes"],
    d_model    = ckpt["d_model"],
    n_heads    = ckpt["n_heads"],
    n_layers   = ckpt["n_layers"],
    d_ff       = ckpt["d_ff"],
)
maskgit.load_state_dict(ckpt["model_state"])
maskgit.eval()

# ── Load VQVAE ────────────────────────────────────────────────────────────────
print("Loading VQVAE...")
vqvae = VQVAE(num_embeddings=128, embedding_dim=64, in_channels=1)
vqvae.load_state_dict(
    torch.load(os.path.join(BASE_DIR, "vqvae_mnist.pt"), map_location="cpu")
)
vqvae.eval()


# ── Figure 1: 4 samples per digit, all 10 digits ─────────────────────────────
print("Generating 4 samples per digit...")
labels    = [d for d in range(10) for _ in range(4)]  # [0,0,0,0,1,1,1,1,...]
sequences = generate_tokens(maskgit, labels, T=20, temperature=1.0, cfg_weight=3.0)
images    = decode_tokens(vqvae, sequences)

fig, axes = plt.subplots(10, 4, figsize=(6, 14))
fig.suptitle("Class-Conditional MaskGIT\n4 samples per digit (0–9)", fontsize=13)
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i].squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    if i % 4 == 0:
        ax.set_ylabel(str(labels[i]), fontsize=14, rotation=0,
                      labelpad=20, va="center")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "mnist_conditional.png"), dpi=150)
plt.show()
print("Saved mnist_conditional.png")


# ── Figure 2: Effect of CFG weight ───────────────────────────────────────────
print("Testing CFG weights...")
fig, axes = plt.subplots(5, 8, figsize=(14, 9))
fig.suptitle("Effect of CFG guidance weight w\n"
             "w=0: unconditional  |  w=3: conditional  |  w=7: strong conditioning",
             fontsize=12)

cfg_weights = [0, 1, 3, 5, 7]
for row, w in enumerate(cfg_weights):
    seqs = generate_tokens(maskgit, [1]*8, T=20, temperature=1.0, cfg_weight=w)
    imgs = decode_tokens(vqvae, seqs)
    for col in range(8):
        axes[row, col].imshow(imgs[col].squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[row, col].axis("off")
    axes[row, 0].set_ylabel(f"w={w}", fontsize=11, rotation=0,
                             labelpad=30, va="center")

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "mnist_cfg_weight.png"), dpi=150)
plt.show()
print("Saved mnist_cfg_weight.png")