import torch
import torch.nn as nn
import math



import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, seq_len: int,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model   = d_model
        self.token_emb = nn.Embedding(vocab_size + 1, d_model)  # +1 for MASK

        pe = self._build_sinusoidal(seq_len, d_model)
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(dropout)

    def _build_sinusoidal(self, seq_len, d_model):
        pe  = torch.zeros(seq_len, d_model)
        pos = torch.arange(seq_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(tokens)
        x = x + self.pe
        return self.dropout(x)


# then MaskGITTransformer follows below...
class MaskGITTransformer(nn.Module):
    def __init__(
        self,
        vocab_size : int,
        seq_len    : int,
        d_model    : int,
        n_heads    : int,
        n_layers   : int,
        d_ff       : int,
        n_classes  : int   = 10,    # 0 = unconditional, 10 = MNIST
        dropout    : float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len    = seq_len
        self.MASK_ID    = vocab_size
        self.n_classes  = n_classes
        self.NULL_CLASS = n_classes  # extra slot for dropped labels

        self.embedding = TokenEmbedding(vocab_size, d_model, seq_len, dropout)

        # Label conditioning — only if n_classes > 0
        if n_classes > 0:
            self.label_emb = nn.Embedding(n_classes + 1, d_model)
        else:
            self.label_emb = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, norm=nn.LayerNorm(d_model),
        )
        self.head = nn.Linear(d_model, vocab_size + 1)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, masked_tokens: torch.Tensor,
                labels: torch.Tensor = None,
                drop_label: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            masked_tokens : (B, L)
            labels        : (B,) — digit class 0-9, None if unconditional
            drop_label    : (B,) bool — True = use NULL_CLASS (CFG dropout)
        """
        x = self.embedding(masked_tokens)              # (B, L, d_model)

        # Add label conditioning if provided
        if self.label_emb is not None and labels is not None:
            if drop_label is not None:
                labels = torch.where(
                    drop_label,
                    torch.full_like(labels, self.NULL_CLASS),
                    labels
                )
            label_vec = self.label_emb(labels)         # (B, d_model)
            x = x + label_vec.unsqueeze(1)             # broadcast over L

        x      = self.transformer(x)
        logits = self.head(x)
        return logits