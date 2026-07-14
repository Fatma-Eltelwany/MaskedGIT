import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from mnist_data import get_dataloader
from model import MaskGITTransformer
from utils import get_mask

# ── Output paths ──────────────────────────────────────────────────────────────
OUTPUT_DIR      = os.environ.get("OUTPUT_DIR", "checkpoints")
os.makedirs(OUTPUT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "maskgit_mnist.pt")


def train(
    vocab_size  : int   = 8,      # 8 grey levels
    seq_len     : int   = 784,    # 28×28
    d_model     : int   = 256,    # bigger — longer sequences need more capacity
    n_heads     : int   = 8,      # more heads for 256-dim
    n_layers    : int   = 6,      # deeper
    d_ff        : int   = 1024,
    dropout     : float = 0.1,
    n_epochs    : int   = 50,     # 60k sequences — each epoch sees much more data
    batch_size  : int   = 128,    # larger batch for stability
    lr          : float = 3e-4,   # standard for transformers
    task        : str   = "mnist",
    sched_mode  : str   = "arccos",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on : {device}")
    print(f"Task        : {task}")

    # ── Data ──────────────────────────────────────────────────────────────────
    loader = get_dataloader(
    task=task, batch_size=batch_size,
    n_bins=vocab_size, data_root="./mnist_data")
    # ── Model ─────────────────────────────────────────────────────────────────
    model = MaskGITTransformer(
        vocab_size=vocab_size,
        seq_len=seq_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
    ).to(device)
    print(f"Parameters  : {sum(p.numel() for p in model.parameters()):,}")

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    # AdamW with weight decay matches the reference implementation
    optimiser = AdamW(
        model.parameters(), lr=lr,
        betas=(0.9, 0.999), weight_decay=0.03
    )
    scheduler = CosineAnnealingLR(optimiser, T_max=n_epochs)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss    = 0.0
        total_correct = 0
        total_masked  = 0
        total_recoverable_correct = 0   # ← add
        total_recoverable         = 0   # ← add
        for tokens in loader:
            tokens = tokens.to(device)               # (B, L)

            # ── Step 1: Mask ───────────────────────────────────────────────
            # Replace masked positions with MASK token ID (= vocab_size)
            # mask is True at positions that are masked
            masked_tokens, mask = get_mask(tokens, mode=sched_mode)

            # Skip batch if nothing got masked (rare with arccos schedule)
            if not mask.any():
                continue

            # ── Step 2: Forward pass ───────────────────────────────────────
            # Model only sees masked_tokens — masked positions are vocab_size
            logits = model(masked_tokens)            # (B, L, vocab_size+1)

            # ── Step 3: Loss ───────────────────────────────────────────────
            # Paper: loss only on masked positions.
            # We implement this by setting unmasked targets to -100,
            # which cross_entropy ignores automatically.
            target = tokens.clone()
            target[~mask] = -100                     # ignore unmasked positions

            loss = nn.functional.cross_entropy(
                logits.reshape(-1, vocab_size + 1),  # (B*L, vocab_size+1)
                target.reshape(-1),                  # (B*L,)
                ignore_index=-100
            )

            # ── Step 4: Backward pass ──────────────────────────────────────
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            # ── Metrics (masked positions only) ────────────────────────────
            with torch.no_grad():
                # Standard masked accuracy (what we had before)
                masked_logits  = logits[mask]
                masked_targets = tokens[mask]
                total_correct += (masked_logits.argmax(-1) == masked_targets).sum().item()
                total_masked  += mask.sum().item()
                total_loss    += loss.item() * mask.sum().item()

                # ── Better metric: recoverable accuracy ───────────────────────────
                # For copy task: a masked position is "recoverable" if its mirror
                # position is visible. Only measure accuracy on those positions.
                # For constant task: any visible token tells you everything,
                # so all masked positions are recoverable if ANY token is visible.
                if task == "copy":
                    half = seq_len // 2
                    # mirror of position i is i+half (or i-half if i>=half)
                    mirror_visible = torch.zeros_like(mask)
                    mirror_visible[:, :half]  = ~mask[:, half:]   # left recoverable if right visible
                    mirror_visible[:, half:]  = ~mask[:, :half]   # right recoverable if left visible
                    recoverable   = mask & mirror_visible          # masked AND mirror visible
                    if recoverable.any():
                        rec_logits  = logits[recoverable]
                        rec_targets = tokens[recoverable]
                        total_recoverable_correct += (rec_logits.argmax(-1) == rec_targets).sum().item()
                        total_recoverable         += recoverable.sum().item()
        scheduler.step()

        # ── Logging ───────────────────────────────────────────────────────
        if epoch % 10 == 0 or epoch == 1:
            avg_loss = total_loss / max(total_masked, 1)
            accuracy = total_correct / max(total_masked, 1)
            
            if task == "copy" and total_recoverable > 0:
                rec_accuracy = total_recoverable_correct / total_recoverable
                print(
                    f"Epoch {epoch:>4}/{n_epochs}"
                    f"  loss: {avg_loss:.4f}"
                    f"  acc: {accuracy:.3f}"
                    f"  recoverable_acc: {rec_accuracy:.3f}"  # ← this is the real metric
                    f"  lr: {scheduler.get_last_lr()[0]:.2e}"
                )
            else:
                print(
                    f"Epoch {epoch:>4}/{n_epochs}"
                    f"  loss: {avg_loss:.4f}"
                    f"  acc:  {accuracy:.3f}"
                    f"  lr:   {scheduler.get_last_lr()[0]:.2e}"
                )
    # ── Save checkpoint ───────────────────────────────────────────────────────
    
    checkpoint = {
        "model_state" : model.state_dict(),
        "vocab_size"  : vocab_size,
        "seq_len"     : seq_len,
        "d_model"     : d_model,
        "n_heads"     : n_heads,
        "n_layers"    : n_layers,
        "d_ff"        : d_ff,
        "task"        : task,
        "final_acc"   : accuracy,   # log what accuracy we ended at
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"\nSaved checkpoint → {CHECKPOINT_PATH}")
    return model


if __name__ == "__main__":
    train()