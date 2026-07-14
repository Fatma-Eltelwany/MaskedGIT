import os
import torch
import torch._dynamo
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from cmnist_vqvae_data import get_dataloader
from cmodel import MaskGITTransformer
from utils import get_mask

OUTPUT_DIR      = os.environ.get("OUTPUT_DIR", "checkpoints")
os.makedirs(OUTPUT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "maskgit_cond.pt")


def train(
    vocab_size      : int   = 128,
    seq_len         : int   = 49,
    n_classes       : int   = 10,    # ← NEW
    d_model         : int   = 128,
    n_heads         : int   = 4,
    n_layers        : int   = 4,
    d_ff            : int   = 512,
    dropout         : float = 0.1,
    n_epochs        : int   = 100,
    batch_size      : int   = 128,
    lr              : float = 3e-4,
    sched_mode      : str   = "arccos",
    label_drop_prob : float = 0.1,   # ← NEW: 10% of batches drop the label
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on : {device}")
    print(f"Conditional : {n_classes} classes, drop_prob={label_drop_prob}")

    loader = get_dataloader(batch_size=batch_size)

    model = MaskGITTransformer(
        vocab_size=vocab_size, seq_len=seq_len,
        n_classes=n_classes,             # ← NEW
        d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ff=d_ff, dropout=dropout,
    ).to(device)
    print(f"Parameters  : {sum(p.numel() for p in model.parameters()):,}")

    optimiser = AdamW(model.parameters(), lr=lr,
                      betas=(0.9, 0.999), weight_decay=0.03)
    scheduler = CosineAnnealingLR(optimiser, T_max=n_epochs)

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = total_correct = total_masked = 0

        for tokens, labels in loader:        # ← unpack tuple
            tokens = tokens.to(device)
            labels = labels.to(device)

            masked_tokens, mask = get_mask(tokens, mode=sched_mode)
            if not mask.any():
                continue

            # CFG: randomly drop label for some sequences each batch
            # model learns both p(image|class) and p(image) simultaneously
            drop_label = torch.rand(len(tokens), device=device) < label_drop_prob

            logits = model(masked_tokens, labels, drop_label)   # (B, L, vocab+1)

            target = tokens.clone()
            target[~mask] = -100
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, vocab_size + 1),
                target.reshape(-1),
                ignore_index=-100
            )

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            with torch.no_grad():
                masked_logits  = logits[mask]
                masked_targets = tokens[mask]
                total_correct += (masked_logits.argmax(-1) == masked_targets).sum().item()
                total_masked  += mask.sum().item()
                total_loss    += loss.item() * mask.sum().item()

        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            avg_loss = total_loss / max(total_masked, 1)
            accuracy = total_correct / max(total_masked, 1)
            print(f"Epoch {epoch:>4}/{n_epochs}"
                  f"  loss: {avg_loss:.4f}"
                  f"  acc:  {accuracy:.3f}"
                  f"  lr:   {scheduler.get_last_lr()[0]:.2e}")

    checkpoint = {
        "model_state" : model.state_dict(),
        "vocab_size"  : vocab_size,
        "seq_len"     : seq_len,
        "n_classes"   : n_classes,
        "d_model"     : d_model,
        "n_heads"     : n_heads,
        "n_layers"    : n_layers,
        "d_ff"        : d_ff,
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"\nSaved → {CHECKPOINT_PATH}")
    return model


if __name__ == "__main__":
    train()