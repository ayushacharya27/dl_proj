import torch
import torch.nn as nn


class PollutionEncoder(nn.Module):
    """
    Encodes pollution features at every
    timestep and station.

    Input:
        [B, T, N, pollution_features]

    Output:
        [B, T, N, embedding_dim]
    """

    def __init__(
        self,
        input_dim=6,
        embedding_dim=64,
        dropout=0.1
    ):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(
                input_dim,
                embedding_dim
            ),

            nn.LayerNorm(
                embedding_dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            )
        )

    def forward(self, x):

        return self.projection(x)


if __name__ == "__main__":

    # Dummy batch
    x = torch.randn(
        32,
        72,
        12,
        6
    )

    model = PollutionEncoder(
        input_dim=6,
        embedding_dim=64
    )

    output = model(x)

    print("Input shape: ")
    print(x.shape)

    print("\nOutput shape:")
    print(output.shape)

    print(
        "\nParameters:",
        sum(
            p.numel()
            for p in model.parameters()
        )
    )