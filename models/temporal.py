import torch
import torch.nn as nn


class TemporalTransformer(nn.Module):
    """
    Temporal Transformer.

    Input:
        [B, T, N, D]

    Output:
        [B, T, N, D]

    Self-attention is applied across T,
    independently for every station.
    """

    def __init__(
        self,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        max_seq_length=72
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        # Learnable temporal positional embeddings.
        self.positional_embedding = nn.Parameter(
            torch.randn(
                1,
                max_seq_length,
                1,
                embedding_dim
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(
            embedding_dim
        )

    def forward(self, x):
        """
        x:
            [B, T, N, D]
        """

        B, T, N, D = x.shape

        # Add temporal positional information.
        x = (
            x
            + self.positional_embedding[:, :T]
        )

        # Treat every station as an independent
        # temporal sequence.
        #
        # [B, T, N, D]
        #      ↓
        # [B*N, T, D]

        x = x.permute(
            0,
            2,
            1,
            3
        )

        x = x.reshape(
            B * N,
            T,
            D
        )

        x = self.transformer(x)

        x = self.norm(x)

        # Restore original structure.
        x = x.reshape(
            B,
            N,
            T,
            D
        )

        x = x.permute(
            0,
            2,
            1,
            3
        )

        return x


if __name__ == "__main__":

    x = torch.randn(
        32,
        72,
        12,
        64
    )

    model = TemporalTransformer(
        embedding_dim=64,
        num_heads=4,
        num_layers=2
    )

    output = model(x)

    print(
        "Input:",
        x.shape
    )

    print(
        "Output:",
        output.shape
    )

    print(
        "Parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        )
    )