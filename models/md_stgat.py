import torch
import torch.nn as nn

from models.encoders import PollutionEncoder
from models.weather_encoder import WeatherEncoder
from models.gat import SpatialGAT
from models.cross_attention import CrossModalAttention
from models.temporal import TemporalTransformer


class MDSTGAT(nn.Module):
    """
    Multi-Modal Decoupled Spatio-Temporal Graph Attention Network

    Pipeline:
        Pollution
            -> Pollution Encoder
            -> Pollution GAT
                           \
                            Cross Attention
                           /
        Weather
            -> Weather Encoder
            -> Weather GAT

        -> Temporal Transformer
        -> 24-hour PM2.5 forecast
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

        self.hidden_dim = hidden_dim
        self.num_stations = num_stations
        self.forecast_horizon = forecast_horizon

        # ---------------------------------------------------------
        # 1. Separate modality encoders
        # ---------------------------------------------------------
        self.pollution_encoder = PollutionEncoder(
            input_dim=pollution_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.weather_encoder = WeatherEncoder(
            input_dim=weather_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # ---------------------------------------------------------
        # 2. Separate spatial GATs
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # 3. Cross-modal attention (Weather = Query, Pollution = Key/Value)
        # ---------------------------------------------------------
        self.cross_attention = CrossModalAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
        )

        # ---------------------------------------------------------
        # 4. Temporal Transformer
        # ---------------------------------------------------------
        self.temporal_transformer = TemporalTransformer(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            dropout=dropout,
        )

        # ---------------------------------------------------------
        # 5. Forecast head
        # ---------------------------------------------------------
        self.forecast_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, forecast_horizon),
        )

    def forward(self, pollution, weather, edge_index):
        """
        Parameters
        ----------
        pollution : Tensor
            Shape [B, T, N, 6]
        weather : Tensor
            Shape [B, T, N, 7]
        edge_index : Tensor
            Shape [2, E]

        Returns
        -------
        forecast : Tensor
            Shape [B, H, N]
        attention : Tensor
            Shape [B, N, T, T]
        """
        # 1. Encode modalities independently -> [B, T, N, D]
        pollution_features = self.pollution_encoder(pollution)
        weather_features = self.weather_encoder(weather)

        # 2. Spatial graph attention independently -> [B, T, N, D]
        pollution_spatial = self.pollution_gat(pollution_features, edge_index)
        weather_spatial = self.weather_gat(weather_features, edge_index)

        # 3. Cross-modal temporal attention -> [B, T, N, D], [B, N, T, T]
        fused_features, attention = self.cross_attention(
            pollution_spatial,
            weather_spatial,
        )

        # 4. Temporal modeling -> [B, T, N, D]
        temporal_features = self.temporal_transformer(fused_features)

        # 5. Extract final timestep representation -> [B, N, D]
        last_state = temporal_features[:, -1, :, :]

        # 6. Multi-horizon forecast -> [B, N, H] -> [B, H, N]
        forecast = self.forecast_head(last_state).permute(0, 2, 1)

        return forecast, attention


# =============================================================
# Standalone test
# =============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Dummy inputs
    B, T, N = 4, 72, 12
    pollution = torch.randn(B, T, N, 6, device=device)
    weather = torch.randn(B, T, N, 7, device=device)

    # 4-nearest-neighbor graph (12 stations × 4 neighbors = 48 directed edges)
    edge_index = torch.randint(0, N, (2, 48), device=device)

    # Model initialization
    model = MDSTGAT(
        pollution_dim=6,
        weather_dim=7,
        hidden_dim=64,
        num_stations=12,
        forecast_horizon=24,
    ).to(device)

    # Forward pass
    with torch.no_grad():
        forecast, attention = model(pollution, weather, edge_index)

    # Results
    print("\nPollution input:", pollution.shape)
    print("Weather input:", weather.shape)
    print("Forecast:", forecast.shape)
    print("Attention:", attention.shape)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Parameters:", total_params)