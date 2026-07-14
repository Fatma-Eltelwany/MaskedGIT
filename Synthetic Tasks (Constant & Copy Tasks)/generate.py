import torch
import torch.nn.functional as F
import os

from model import MaskGITTransformer
from utils import cosine_schedule, mask_by_confidence


@torch.no_grad()
def generate(
    model      : MaskGITTransformer,
    batch_size : int   = 4,
    T          : int   = 10,
    temperature: float = 1.0,
    seed       : int   = 42,
    verbose    : bool  = True,
) -> tuple[torch.Tensor, list]:
    """
    MaskGIT iterative decoding — Algorithm 1 in the paper.

    Step 0 : start fully masked (all tokens = MASK_ID)
    For t = 1 … T:
        1. Predict  — transformer gives p(y_i | visible tokens)
        2. Sample   — draw y_i ~ p_i, record confidence
        3. Schedule — n = floor(γ(t/T) · N) tokens still masked
        4. Mask     — re-mask n least-confident positions

    Paper connection:
        The MASK_ID token (= vocab_size) is what the model sees at
        masked positions. It was trained on this exact setup, so at
        inference we just start with all MASK_IDs and let it fill in.
    """
    torch.manual_seed(seed)
    device   = next(model.parameters()).device
    seq_len  = model.seq_len
    MASK_ID  = model.MASK_ID     # = vocab_size

    # ── Step 0: start fully masked ────────────────────────────────────────────
    tokens = torch.full(
        (batch_size, seq_len), MASK_ID,
        dtype=torch.long, device=device
    )

    if verbose:
        print(f"Generating {batch_size} sequences | T={T} steps | "
              f"seq_len={seq_len} | vocab_size={MASK_ID}\n")
        _print_state(tokens, mask=None, step=0, T=T, seq_len=seq_len, MASK_ID=MASK_ID)

    history = [tokens.clone()]

    # ── Iterative decoding ────────────────────────────────────────────────────
    for t in range(1, T + 1):

        # 1. PREDICT — forward pass
        logits = model(tokens)                          # (B, L, vocab_size+1)

        # Slice off the mask token logit — we never want to sample MASK_ID
        logits = logits[:, :, :MASK_ID]                # (B, L, vocab_size)

        # Apply temperature
        logits = logits / temperature

        # 2. SAMPLE — draw token at every position
        probs   = F.softmax(logits, dim=-1)            # (B, L, vocab_size)
        sampled = torch.multinomial(
            probs.view(-1, MASK_ID), num_samples=1
        ).view(batch_size, seq_len)                    # (B, L)

        # Confidence = probability of the sampled token
        confidence = probs.gather(
            dim=-1, index=sampled.unsqueeze(-1)
        ).squeeze(-1)                                  # (B, L)

        # Only update currently-masked positions
        currently_masked = (tokens == MASK_ID)
        tokens     = torch.where(currently_masked, sampled, tokens)

        # Revealed positions get confidence=1 so they're never re-masked
        confidence = torch.where(
            currently_masked, confidence, torch.ones_like(confidence)
        )

        # 3. SCHEDULE — how many tokens to keep masked after this step?
        n_to_mask = cosine_schedule(t, T, seq_len)     # paper: n=⌊γ(t/T)·N⌋

        # 4. MASK — re-mask the n least-confident positions
        new_mask = mask_by_confidence(confidence, n_to_mask)
        tokens[new_mask] = MASK_ID

        history.append(tokens.clone())

        if verbose:
            _print_state(tokens, new_mask, step=t, T=T,
                         seq_len=seq_len, MASK_ID=MASK_ID)

    return tokens, history


def _print_state(tokens, mask, step, T, seq_len, MASK_ID):
    """Pretty-print one sequence to visualise decoding progress."""
    seq      = tokens[0].tolist()
    n_masked = sum(1 for x in seq if x == MASK_ID)
    half     = seq_len // 2

    left  = ["  ?" if x == MASK_ID else f"{x:3d}" for x in seq[:half]]
    right = ["  ?" if x == MASK_ID else f"{x:3d}" for x in seq[half:]]

    print(f"Step {step:>2}/{T}  masked={n_masked:>2}/{seq_len}"
          f"  [{' '.join(left)} |{' '.join(right)}]")


def verify_copy(tokens: torch.Tensor) -> torch.Tensor:
    """
    Check which generated sequences perfectly satisfy the copy rule.
    Returns bool tensor of shape (batch,).
    """
    half  = tokens.shape[1] // 2
    left  = tokens[:, :half]
    right = tokens[:, half:]
    return (left == right).all(dim=1)


if __name__ == "__main__":
    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "checkpoints")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt  = torch.load(
        os.path.join(OUTPUT_DIR, "maskgit.pt"), map_location="cpu"
    )
    model = MaskGITTransformer(
        vocab_size = ckpt["vocab_size"],
        seq_len    = ckpt["seq_len"],
        d_model    = ckpt["d_model"],
        n_heads    = ckpt["n_heads"],
        n_layers   = ckpt["n_layers"],
        d_ff       = ckpt["d_ff"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ── Generate ──────────────────────────────────────────────────────────────
    sequences, history = generate(
        model, batch_size=4, T=20, temperature=0.7
    )

    # ── Verify copy rule ──────────────────────────────────────────────────────
    valid = verify_copy(sequences)
    half  = sequences.shape[1] // 2

    print(f"\nCopy rule satisfied: {valid.sum().item()}/{len(valid)} sequences\n")
    print("Generated sequences:")
    for i, seq in enumerate(sequences):
        mark = "✓" if valid[i] else "✗"
        left  = seq[:half].tolist()
        right = seq[half:].tolist()
        print(f"  {mark}  {left} | {right}")