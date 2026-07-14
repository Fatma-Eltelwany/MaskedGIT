import torch
from torch.utils.data import Dataset, DataLoader


class ConstantDataset(Dataset):
    """
    Every sequence is a constant repetition of one token value.

        sequence = [v, v, v, ..., v]   where v ∈ {0, ..., vocab_size-1}

    MaskGIT connection:
        Revealing ANY single token tells the model everything.
        This is the purest test of bidirectional attention —
        the model should learn "all tokens in this sequence are the same"
        and achieve perfect accuracy almost immediately.

        Analogy to images: like a uniformly coloured image patch —
        seeing one pixel tells you all pixels.
    """

    def __init__(self, vocab_size: int = 16, seq_len: int = 20):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len    = seq_len

        # One sequence per token value → vocab_size sequences total
        self.sequences = torch.stack([
            torch.full((seq_len,), v, dtype=torch.long)
            for v in range(vocab_size)
        ])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


class CopyDataset(Dataset):
    """
    Each sequence has two halves:
        left  half: random tokens
        right half: exact copy of left half

        e.g. seq_len=20 → [3,7,1,9,2,5,8,0,4,6 | 3,7,1,9,2,5,8,0,4,6]

    MaskGIT connection:
        Information flows BOTH ways across the sequence:
        - Mask right half → predict from left half
        - Mask left half  → predict from right half
        - Mask both partially → combine evidence from both sides

        This is fundamentally bidirectional — no causal direction.
        Analogy to images: like an image with a symmetry axis —
        any patch on one side predicts the mirror patch.

        All possible left halves → vocab_size^(seq_len//2) sequences.
        We sample a large random subset rather than enumerate all.
    """

    def __init__(
        self,
        vocab_size  : int = 16,
        seq_len     : int = 20,
        n_sequences : int = 2048,
        seed        : int = 42,
    ):
        super().__init__()
        assert seq_len % 2 == 0, "seq_len must be even for copy task"
        self.vocab_size = vocab_size
        self.seq_len    = seq_len
        half            = seq_len // 2

        torch.manual_seed(seed)
        # Sample random left halves
        left  = torch.randint(0, vocab_size, (n_sequences, half))
        # Right half is an exact copy
        right = left.clone()
        self.sequences = torch.cat([left, right], dim=1)  # (N, seq_len)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def get_dataloader(
    task        : str  = "copy",   # "constant" or "copy"
    vocab_size  : int  = 16,
    seq_len     : int  = 20,
    n_sequences : int  = 2048,     # only used for copy task
    batch_size  : int  = 64,
    shuffle     : bool = True,
):
    if task == "constant":
        dataset = ConstantDataset(vocab_size, seq_len)
    elif task == "copy":
        dataset = CopyDataset(vocab_size, seq_len, n_sequences)
    else:
        raise ValueError(f"Unknown task: {task}. Choose 'constant' or 'copy'.")

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ── Sanity checks ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 55)
    print("CONSTANT TASK")
    print("=" * 55)
    ds = ConstantDataset(vocab_size=16, seq_len=20)
    print(f"Dataset size: {len(ds)} sequences\n")
    for i in range(4):
        seq = ds[i].tolist()
        valid = len(set(seq)) == 1   # all tokens identical
        print(f"  {'✓' if valid else '✗'}  {seq}")

    print()
    print("=" * 55)
    print("COPY TASK")
    print("=" * 55)
    ds = CopyDataset(vocab_size=16, seq_len=20, n_sequences=2048)
    print(f"Dataset size: {len(ds)} sequences\n")
    for i in range(4):
        seq   = ds[i].tolist()
        left  = seq[:10]
        right = seq[10:]
        valid = left == right
        print(f"  {'✓' if valid else '✗'}  {left} | {right}")

    print()
    print("=" * 55)
    print("DATALOADER")
    print("=" * 55)
    loader = get_dataloader(task="copy", batch_size=64)
    batch  = next(iter(loader))
    print(f"Batch shape : {batch.shape}")       # (64, 20)
    print(f"Left half   : {batch[0, :10].tolist()}")
    print(f"Right half  : {batch[0, 10:].tolist()}")
    print(f"Match       : {batch[0,:10].tolist() == batch[0,10:].tolist()}")