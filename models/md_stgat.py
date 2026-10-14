
import torch
import torch.nn as nn

from models.encoders import PollutionEncoder
from models.weather_encoder import WeatherEncoder
from models.gat import SpatialGAT
from models.cross_attention import CrossModalAttention
from models.temporal import TemporalTransformer


class MDSTGAT(nn.Module):
    """
    Multi-Modal Decoupled Spatio-Temporal Graph Attention Network.

    Inputs:
        pollution: [B, T, N, 6]
        weather:   [B, T, N, 7]
        edge_index: [2, E]

    Outputs:
        forecast:  [B, H, N]
        attention: [B, N, T, T]
    """

    def __init__(
        self,
        pollution_dim=6,
        weather_dim=7,
        hidden_dim=64,
        num_stations=12,
        forecast_horizon=24,
        gat_heads=4,
        transformer_heads=4,
        transformer_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        if hidden_dim % gat_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by gat_heads."
            )

        if hidden_dim % transformer_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by transformer_heads."
            )

        self.hidden_dim = hidden_dim
        self.num_stations = num_stations
        self.forecast_horizon = forecast_horizon

        # 1. Independent modality encoders
        self.pollution_encoder = PollutionEncoder(
            input_dim=pollution_dim,
            embedding_dim=hidden_dim,
            dropout=dropout,
        )

        self.weather_encoder = WeatherEncoder(
            input_dim=weather_dim,
            embedding_dim=hidden_dim,
            dropout=dropout,
        )

        # 2. Independent spatial graph attention networks
        self.pollution_gat = SpatialGAT(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            heads=gat_heads,
            dropout=dropout,
        )

        self.weather_gat = SpatialGAT(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            heads=gat_heads,
            dropout=dropout,
        )

        # 3. Weather queries historical pollution representations
        self.cross_attention = CrossModalAttention(
            embedding_dim=hidden_dim,
            num_heads=transformer_heads,
            dropout=dropout,
        )

        # 4. Temporal modeling
        self.temporal_transformer = TemporalTransformer(
            embedding_dim=hidden_dim,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            dropout=dropout,
        )

        # 5. Direct multi-horizon forecast head
        self.forecast_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, forecast_horizon),
        )

    def forward(self, pollution, weather, edge_index):
        """
        pollution: [B, T, N, pollution_dim]
        weather:   [B, T, N, weather_dim]
        edge_index: [2, E]

        Returns:
            forecast:  [B, H, N]
            attention: [B, N, T, T]
        """

        if pollution.ndim != 4 or weather.ndim != 4:
            raise ValueError(
                "Pollution and weather must be 4D tensors [B,T,N,F]."
            )

        if pollution.shape[:3] != weather.shape[:3]:
            raise ValueError(
                "Pollution and weather batch, time, and station "
                "dimensions must match."
            )

        if pollution.shape[-1] != self.pollution_encoder.projection[0].in_features:
            raise ValueError("Unexpected pollution feature dimension.")

        if weather.shape[-1] != self.weather_encoder.projection[0].in_features:
            raise ValueError("Unexpected weather feature dimension.")

        B, T, N, _ = pollution.shape

        if N != self.num_stations:
            raise ValueError(
                f"Expected {self.num_stations} stations, received {N}."
            )

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E].")

        edge_index = edge_index.to(
            device=pollution.device,
            dtype=torch.long,
        )

        # 1. Encode each modality independently
        pollution_features = self.pollution_encoder(pollution)
        weather_features = self.weather_encoder(weather)

        # 2. Spatial attention for each modality
        pollution_spatial = self.pollution_gat(
            pollution_features, edge_index
        )

        weather_spatial = self.weather_gat(
            weather_features, edge_index
        )

        # 3. Cross-modal temporal attention
        # Weather = Query; Pollution = Key and Value
        fused_features, attention = self.cross_attention(
            pollution_spatial,
            weather_spatial,
        )

        # 4. Temporal Transformer
        temporal_features = self.temporal_transformer(
            fused_features
        )

        # 5. Last observed timestep representation
        last_state = temporal_features[:, -1, :, :]  # [B,N,D]

        # 6. Predict all 24 future hours directly
        forecast = self.forecast_head(last_state)  # [B,N,H]
        forecast = forecast.permute(0, 2, 1).contiguous()

        return forecast, attention



if __name__ == "__main__":
    from data.graph import build_knn_graph

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    B, T, N = 2, 72, 12

    pollution = torch.randn(B, T, N, 6, device=device)
    weather = torch.randn(B, T, N, 7, device=device)

    # Build the real station graph.
    graph_result = build_knn_graph(k=4)

    # Some graph builders return (edges, additional_information).
    graph_data = (
        graph_result[0]
        if isinstance(graph_result, tuple)
        else graph_result
    )

    edge_index = torch.as_tensor(graph_data, dtype=torch.long)

    print("Original graph shape:", tuple(edge_index.shape))

    if edge_index.ndim != 2:
        raise ValueError(
            f"Graph must be 2D, got {tuple(edge_index.shape)}"
        )

    if edge_index.shape[0] == 2:
        # Already [2, E].
        pass

    elif edge_index.shape[1] == 2:
        # Convert [E, 2] into [2, E].
        edge_index = edge_index.t().contiguous()

    elif edge_index.shape[0] == N:
        # Interpret [N, k] as a list of neighbors for each station.
        neighbors = edge_index

        if neighbors.min() < 0 or neighbors.max() >= N:
            raise ValueError("Neighbor indices are outside the station range.")

        sources = torch.arange(N).repeat_interleave(
            neighbors.shape[1]
        )
        targets = neighbors.reshape(-1)

        edge_index = torch.stack([sources, targets], dim=0)

    else:
        raise ValueError(
            f"Unrecognized graph shape: {tuple(edge_index.shape)}. "
            "Inspect data/graph.py to identify its format."
        )

    if edge_index.min() < 0 or edge_index.max() >= N:
        raise ValueError("Graph contains invalid station indices.")

    edge_index = edge_index.to(device)

    print("Graph shape used by GAT:", tuple(edge_index.shape))

    # Initialize model.
    model = MDSTGAT(
        pollution_dim=6,
        weather_dim=7,
        hidden_dim=64,
        num_stations=N,
        forecast_horizon=24,
    ).to(device)

    model.eval()

    # Test a forward pass.
    with torch.inference_mode():
        forecast, attention = model(
            pollution,
            weather,
            edge_index,
        )

    print("Pollution input:", tuple(pollution.shape))
    print("Weather input:", tuple(weather.shape))
    print("Forecast:", tuple(forecast.shape))
    print("Attention:", tuple(attention.shape))

    total_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Trainable parameters:", total_params)
