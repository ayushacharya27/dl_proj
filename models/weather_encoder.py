import torch
import torch.nn as nn


class WeatherEncoder(nn.Module):
    """
    Encodes meteorological features independently from the pollution modality.

    Input:
        [B, T, N, 7]

    Output:
        [B, T, N, embedding_dim]
    """

    def __init__(self, input_dim=7, embedding_dim=64, dropout=0.1):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.projection(x)


if __name__ == "__main__":
    x = torch.randn(32, 72, 12, 7)
    model = WeatherEncoder(input_dim=7, embedding_dim=64)
    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Parameters:", sum(p.numel() for p in model.parameters()))