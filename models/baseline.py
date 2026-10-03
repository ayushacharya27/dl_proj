import torch
import torch.nn as nn

from models.encoders import PollutionEncoder
from models.gat import SpatialGAT
from models.temporal import TemporalTransformer


class BaselineModel(nn.Module):

    def __init__(
        self,
        pollution_dim=6,
        embedding_dim=64,
        gat_heads=4,
        temporal_heads=4,
        temporal_layers=2,
        forecast_horizon=24,
        num_stations=12,
        dropout=0.1
    ):
        super().__init__()

        self.forecast_horizon = forecast_horizon
        self.num_stations = num_stations

        # ---------------------------------------------
        # Pollution encoder
        # ---------------------------------------------

        self.encoder = PollutionEncoder(
            input_dim=pollution_dim,
            embedding_dim=embedding_dim,
            dropout=dropout
        )

        # ---------------------------------------------
        # Spatial GAT
        # ---------------------------------------------

        self.gat = SpatialGAT(
            input_dim=embedding_dim,
            hidden_dim=embedding_dim,
            heads=gat_heads,
            dropout=dropout
        )

        # ---------------------------------------------
        # Temporal Transformer
        # ---------------------------------------------

        self.temporal = TemporalTransformer(
            embedding_dim=embedding_dim,
            num_heads=temporal_heads,
            num_layers=temporal_layers,
            dropout=dropout,
            max_seq_length=72
        )

        # ---------------------------------------------
        # Forecast head
        # ---------------------------------------------

        self.forecast_head = nn.Sequential(
            nn.Linear(
                embedding_dim,
                embedding_dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                embedding_dim,
                forecast_horizon
            )
        )

    def forward(
        self,
        x,
        edge_index
    ):
        """
        x:
            [B, 72, 12, 6]

        edge_index:
            [2, E]

        returns:
            [B, 24, 12]
        """

        # ---------------------------------------------
        # Encode pollution
        # ---------------------------------------------

        x = self.encoder(x)

        # [B,72,12,64]

        # ---------------------------------------------
        # Spatial reasoning
        # ---------------------------------------------

        x = self.gat(
            x,
            edge_index
        )

        # [B,72,12,64]

        # ---------------------------------------------
        # Temporal reasoning
        # ---------------------------------------------

        x = self.temporal(x)

        # [B,72,12,64]

        # ---------------------------------------------
        # Use final timestep representation
        # ---------------------------------------------

        x = x[:, -1, :, :]

        # [B,12,64]

        # ---------------------------------------------
        # Forecast 24 future hours
        # ---------------------------------------------

        x = self.forecast_head(x)

        # [B,12,24]

        # Rearrange to [B,24,12]

        x = x.permute(
            0,
            2,
            1
        )

        return x


if __name__ == "__main__":

    from data.graph import build_knn_graph

    # ---------------------------------------------
    # Dummy input
    # ---------------------------------------------

    x = torch.randn(
        4,
        72,
        12,
        6
    )

    # ---------------------------------------------
    # Graph
    # ---------------------------------------------

    edges, _ = build_knn_graph(
        k=4
    )

    edge_index = torch.tensor(
        edges.T,
        dtype=torch.long
    )

    # ---------------------------------------------
    # Model
    # ---------------------------------------------

    model = BaselineModel()

    # ---------------------------------------------
    # Forward pass
    # ---------------------------------------------

    output = model(
        x,
        edge_index
    )

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