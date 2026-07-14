
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import torch, os
from model import MaskGITTransformer
from generate_mnist import generate  # or wherever your VQVAE generate function lives
from diversity_check import check_diversity

def check_diversity(sequences: torch.Tensor, vocab_size: int, name: str = "samples"):
    """
    Diagnose mode collapse in a batch of generated sequences.

    Args:
        sequences  : (N, L) generated token sequences
        vocab_size : number of possible token values (excluding MASK)
        name       : label for plot titles
    """
    n_samples, seq_len = sequences.shape

    # ── 1. Exact duplicate check ────────────────────────────────────────────
    seq_strings   = [tuple(s.tolist()) for s in sequences]
    unique_count  = len(set(seq_strings))
    print(f"[{name}] Unique sequences: {unique_count}/{n_samples} "
          f"({unique_count/n_samples:.1%})")

    # ── 2. Per-position entropy ─────────────────────────────────────────────
    position_entropy = np.zeros(seq_len)
    position_top_freq = np.zeros(seq_len)   # frequency of the most common value
    max_entropy = np.log(vocab_size)        # uniform distribution entropy

    for pos in range(seq_len):
        values  = sequences[:, pos].tolist()
        counts  = Counter(values)
        probs   = np.array([c / n_samples for c in counts.values()])
        entropy = -(probs * np.log(probs + 1e-12)).sum()
        position_entropy[pos]  = entropy
        position_top_freq[pos] = max(counts.values()) / n_samples

    avg_entropy_ratio = (position_entropy / max_entropy).mean()
    print(f"[{name}] Avg entropy ratio: {avg_entropy_ratio:.3f}  "
          f"(1.0 = fully uniform/diverse, 0.0 = always same value)")
    print(f"[{name}] Avg top-value frequency: {position_top_freq.mean():.3f}  "
          f"(1.0 = every position always shows same value = total collapse)")

    # ── 3. Visualisation ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Left: entropy per position
    axes[0].plot(position_entropy, color="#C00000", linewidth=1.5)
    axes[0].axhline(max_entropy, color="gray", linestyle="--",
                    label=f"Max entropy (uniform) = {max_entropy:.2f}")
    axes[0].set_xlabel("Token position")
    axes[0].set_ylabel("Entropy (nats)")
    axes[0].set_title(f"Per-Position Entropy — {name}\n"
                      f"Low entropy = position always same value across samples")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # Right: top-value frequency heatmap-style bar
    axes[1].plot(position_top_freq, color="#1E3A5F", linewidth=1.5)
    axes[1].axhline(1/vocab_size, color="gray", linestyle="--",
                    label=f"Random chance = {1/vocab_size:.2f}")
    axes[1].set_xlabel("Token position")
    axes[1].set_ylabel("Frequency of most common value")
    axes[1].set_title(f"Mode Dominance — {name}\n"
                      f"High = one value dominates that position (collapse signal)")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fname = f"diversity_{name.replace(' ', '_')}.png"
    #plt.savefig(fname, dpi=150)
    plt.show()
    #print(f"Saved {fname}\n")

    return {
        "unique_ratio"   : unique_count / n_samples,
        "entropy_ratio"  : avg_entropy_ratio,
        "top_freq"       : position_top_freq.mean(),
    }


# ── Run on copy task ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    from model import MaskGITTransformer
    from generate import generate   # copy task generator

    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "checkpoints")
    ckpt = torch.load(os.path.join(OUTPUT_DIR, "maskgit_copy.pt"), map_location="cpu")
    model = MaskGITTransformer(
        vocab_size=ckpt["vocab_size"], seq_len=ckpt["seq_len"],
        d_model=ckpt["d_model"], n_heads=ckpt["n_heads"],
        n_layers=ckpt["n_layers"], d_ff=ckpt["d_ff"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Generate WITHOUT a fixed seed so each sample is genuinely independent
    torch.manual_seed(torch.seed())  # randomize
    sequences, _ = generate(model, batch_size=200, T=20, temperature=1.0,
                            seed=int(torch.randint(0, 1_000_000, (1,)).item()),
                            verbose=False)

    results = check_diversity(sequences, vocab_size=ckpt["vocab_size"], name="Copy Task")

    print("Interpretation:")
    if results["unique_ratio"] > 0.9 and results["entropy_ratio"] > 0.5:
        print("  ✓ Good diversity — no mode collapse detected.")
    elif results["unique_ratio"] < 0.3 or results["top_freq"] > 0.8:
        print("  ✗ Likely mode collapse — most samples are near-identical.")
    else:
        print("  ~ Mixed signal — some positions diverse, others dominated.")