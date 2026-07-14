import math
import torch

# Add this constant — the mask token is a special index BEYOND the vocabulary
# e.g. vocab_size=16 → MASK_TOKEN_ID=16


def get_mask(tokens: torch.Tensor, mode: str = "arccos") -> tuple[torch.Tensor, torch.Tensor]:
    B, L   = tokens.shape
    device = tokens.device
    MASK_ID = tokens.max().item() + 1  # vocab_size

    r = torch.rand(B, device=device)

    if mode == "arccos":
        val_to_mask = torch.arccos(r) / (math.pi * 0.5)
    elif mode == "cosine":
        val_to_mask = torch.cos(r * math.pi * 0.5)
    elif mode == "linear":
        val_to_mask = 1 - r
    elif mode == "square":
        val_to_mask = 1 - r ** 2
    elif mode == "root":
        val_to_mask = 1 - r ** 0.5
    else:
        raise ValueError(f"Unknown mode: {mode}")

    noise         = torch.rand(B, L, device=device)
    mask          = noise < val_to_mask.unsqueeze(1)
    masked_tokens = tokens.clone()
    masked_tokens[mask] = MASK_ID

    return masked_tokens, mask


def cosine_schedule(t: int, T: int, N: int) -> int:
    """
    Number of tokens to keep masked at decoding step t.
    γ(r) = cos(πr/2),  r = t/T
    """
    r     = t / T
    gamma = math.cos(math.pi * r / 2)
    return math.floor(gamma * N)


def mask_by_confidence(confidence: torch.Tensor, n_to_mask: int) -> torch.Tensor:
    """Re-mask the n least confident positions."""
    B, L = confidence.shape
    if n_to_mask <= 0:
        return torch.zeros(B, L, dtype=torch.bool, device=confidence.device)
    if n_to_mask >= L:
        return torch.ones(B, L, dtype=torch.bool, device=confidence.device)
    sorted_conf, _ = torch.sort(confidence, dim=-1)
    threshold      = sorted_conf[:, n_to_mask - 1].unsqueeze(1)
    return confidence <= threshold