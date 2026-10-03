import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class SpatialGAT(nn.Module):
    """
    Spatial Graph Attention Network.

    Input:
        [B, T, N, D]

    Output:
        [B, T, N, D]

    The same spatial graph is applied independently
    at every timestep.
    """

    def __init__(
        self,
        input_dim=64,
        hidden_dim=64,
        heads=4,
        dropout=0.1
    ):
        super().__init__()

        self.gat = GATConv(
            in_channels=input_dim,
            out_channels=hidden_dim // heads,
            heads=heads,
            concat=True,
            dropout=dropout
        )

        self.norm = nn.LayerNorm(
            hidden_dim
        )

        self.activation = nn.GELU()

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x,
        edge_index
    ):
        """
        x:
            [B, T, N, D]

        edge_index:
            [2, E]
        """

        B, T, N, D = x.shape

        # Flatten batch and temporal dimensions.
        # Each station becomes a graph node.
        x = x.reshape(
            B * T * N,
            D
        )

        # Repeat graph edges for every
        # batch × timestep graph.
        E = edge_index.shape[1]

        offsets = (
            torch.arange(
                B * T,
                device=x.device
            )
            * N
        )

        edge_index = (
            edge_index.unsqueeze(0)
            + offsets[:, None, None]
        )

        edge_index = edge_index.permute(
            1, 2, 0
        ).reshape(
            2,
            E * B * T
        )

        x = self.gat(
            x,
            edge_index
        )

        x = self.activation(x)

        x = self.dropout(x)

        x = x.reshape(
            B,
            T,
            N,
            -1
        )

        x = self.norm(x)

        return x


if __name__ == "__main__":

    from data.graph import build_knn_graph

    # --------------------------------------------------
    # Dummy input
    # --------------------------------------------------

    x = torch.randn(
        32,
        72,
        12,
        64
    )

    # --------------------------------------------------
    # Build graph
    # --------------------------------------------------

    edges, _ = build_knn_graph(
        k=4
    )

    edge_index = torch.tensor(
        edges.T,
        dtype=torch.long
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = SpatialGAT(
        input_dim=64,
        hidden_dim=64,
        heads=4
    )

    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------

    output = model(
        x,
        edge_index
    )

    print(
        "Input:",
        x.shape
    )

    print(
        "Edge index:",
        edge_index.shape
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