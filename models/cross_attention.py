import torch
import torch.nn as nn


class CrossModalAttention(nn.Module):
    """
    Weather-conditioned historical pollution attention.

    Weather provides the Query.
    Historical pollution provides Key and Value.

    For every station:
        current weather -> Query -> attends over -> 72-hour pollution history

    Input:
        pollution: [B, T, N, D]
        weather:   [B, T, N, D]

    Output:
        output:            [B, T, N, D]
        attention_weights: [B, N, T, T]
    """

    def __init__(self, embedding_dim=64, num_heads=4, dropout=0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, pollution, weather):
        """
        pollution: [B, T, N, D]
        weather:   [B, T, N, D]

        Returns:
            output:            [B, T, N, D]
            attention_weights: [B, N, T, T]
        """
        B, T, N, D = pollution.shape

        # --------------------------------------------------
        # Rearrange so each station becomes one sequence:
        # [B, T, N, D] -> [B, N, T, D] -> [B * N, T, D]
        # --------------------------------------------------
        pollution_seq = pollution.permute(0, 2, 1, 3).reshape(B * N, T, D)
        weather_seq = weather.permute(0, 2, 1, 3).reshape(B * N, T, D)

        # --------------------------------------------------
        # Every weather timestep queries the complete pollution history:
        # Q = weather, K = pollution, V = pollution
        # --------------------------------------------------
        attended, attention_weights = self.attention(
            query=weather_seq,
            key=pollution_seq,
            value=pollution_seq,
            need_weights=True
        )

        # Residual connections & feed-forward block
        x = self.norm1(weather_seq + attended)
        x = self.norm2(x + self.feed_forward(x))

        # --------------------------------------------------
        # Restore original tensor layout:
        # [B * N, T, D] -> [B, N, T, D] -> [B, T, N, D]
        # --------------------------------------------------
        x = x.reshape(B, N, T, D).permute(0, 2, 1, 3)

        # Restore station dimension for attention weights:
        # [B * N, T, T] -> [B, N, T, T]
        attention_weights = attention_weights.reshape(B, N, T, T)

        return x, attention_weights


if __name__ == "__main__":
    pollution = torch.randn(4, 72, 12, 64)
    weather = torch.randn(4, 72, 12, 64)

    model = CrossModalAttention(embedding_dim=64, num_heads=4)
    output, attention = model(pollution, weather)

    print("Pollution:", pollution.shape)
    print("Weather:", weather.shape)
    print("Output:", output.shape)
    print("Attention:", attention.shape)
    print("Parameters:", sum(p.numel() for p in model.parameters()))