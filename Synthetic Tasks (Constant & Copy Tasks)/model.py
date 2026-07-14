import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    """
    Now much simpler — no mask logic here.
    The mask token is just a regular token index (vocab_size),
    already substituted into the sequence before this layer.
    Embedding size is vocab_size + 1 to include the mask token.
    """

    def __init__(self, vocab_size: int, d_model: int, seq_len: int,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # vocab_size + 1 to include the MASK token as a real embedding
        self.token_emb = nn.Embedding(vocab_size + 1, d_model)

        pe = self._build_sinusoidal(seq_len, d_model)
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(dropout)

    def _build_sinusoidal(self, seq_len: int, d_model: int) -> torch.Tensor:
        pe  = torch.zeros(seq_len, d_model)
        pos = torch.arange(seq_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens : (B, L) — token indices, masked positions already
                     replaced with vocab_size (the MASK token ID)
        Returns:
            x      : (B, L, d_model)
        """
        x = self.token_emb(tokens)   # (B, L, d_model)
        x = x + self.pe
        return self.dropout(x)


class MaskGITTransformer(nn.Module):

    def __init__(
        self,
        vocab_size : int,
        seq_len    : int,
        d_model    : int,
        n_heads    : int,
        n_layers   : int,
        d_ff       : int,
        dropout    : float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len    = seq_len
        self.MASK_ID    = vocab_size   # mask token index

        self.embedding = TokenEmbedding(vocab_size, d_model, seq_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.head = nn.Linear(d_model, vocab_size+1)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, masked_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            masked_tokens : (B, L) — tokens with masked positions
                            already replaced by self.MASK_ID

        Returns:
            logits : (B, L, vocab_size)

        Note: forward signature is now simpler — no mask argument needed.
        The mask was already applied to the tokens before calling forward.
        """
        x      = self.embedding(masked_tokens)   # (B, L, d_model)
        x      = self.transformer(x)             # (B, L, d_model)
        logits = self.head(x)                    # (B, L, vocab_size)
        return logits