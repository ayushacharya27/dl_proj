import glob
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from data.preprocessing import load_station_file
from data.station_metadata import STATIONS
from data.graph import build_distance_matrix


# ============================================================
# Configuration
# ============================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

INPUT_LENGTH = 72
FORECAST_HORIZON = 24

SHORT_GAP_LIMIT = 6

POLLUTION_FEATURES = [
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
]

WEATHER_FEATURES = [
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM",
    "wd_sin",
    "wd_cos",
]

ALL_NUMERIC = POLLUTION_FEATURES + [
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM",
]


# ============================================================
# Wind direction
# ============================================================

WIND_ANGLES = {
    "N": 0.0,
    "NNE": 22.5,
    "NE": 45.0,
    "ENE": 67.5,
    "E": 90.0,
    "ESE": 112.5,
    "SE": 135.0,
    "SSE": 157.5,
    "S": 180.0,
    "SSW": 202.5,
    "SW": 225.0,
    "WSW": 247.5,
    "W": 270.0,
    "WNW": 292.5,
    "NW": 315.0,
    "NNW": 337.5,
}


def encode_wind_direction(df):
    """
    Convert compass wind direction into sin/cos representation.
    """

    df = df.copy()

    angles = df["wd"].map(WIND_ANGLES)

    radians = np.deg2rad(angles)

    df["wd_sin"] = np.sin(radians)
    df["wd_cos"] = np.cos(radians)

    return df


# ============================================================
# Load data
# ============================================================

def load_data():

    files = sorted(
        glob.glob("data/raw/PRSA_Data_*.csv")
    )

    dataframes = []

    for file in files:

        df = load_station_file(file)

        dataframes.append(df)

    df = pd.concat(
        dataframes,
        ignore_index=True
    )

    df = df.sort_values(
        ["datetime", "station"]
    ).reset_index(drop=True)

    return df


# ============================================================
# Chronological split
# ============================================================

def chronological_split(df):

    timestamps = np.sort(
        df["datetime"].unique()
    )

    n = len(timestamps)

    train_end = int(
        n * TRAIN_RATIO
    )

    val_end = int(
        n * (TRAIN_RATIO + VAL_RATIO)
    )

    train_times = timestamps[:train_end]
    val_times = timestamps[train_end:val_end]
    test_times = timestamps[val_end:]

    train = df[
        df["datetime"].isin(train_times)
    ].copy()

    val = df[
        df["datetime"].isin(val_times)
    ].copy()

    test = df[
        df["datetime"].isin(test_times)
    ].copy()

    print("\nChronological split:")
    print(f"Train: {train['datetime'].min()} → {train['datetime'].max()}")
    print(f"Val:   {val['datetime'].min()} → {val['datetime'].max()}")
    print(f"Test:  {test['datetime'].min()} → {test['datetime'].max()}")

    print("\nRows:")
    print(f"Train: {len(train):,}")
    print(f"Val:   {len(val):,}")
    print(f"Test:  {len(test):,}")

    return train, val, test


# ============================================================
# Causal temporal imputation
# ============================================================

def causal_temporal_imputation(df, max_gap=6):

    df = df.copy()

    for station in STATIONS:

        mask = df["station"] == station

        station_df = (
            df.loc[mask]
            .sort_values("datetime")
            .copy()
        )

        # Only use observations from the past.
        # No interpolation from future values.
        for feature in ALL_NUMERIC:

            station_df[feature] = (
                station_df[feature]
                .ffill(limit=max_gap)
            )

        df.loc[
            mask,
            ALL_NUMERIC
        ] = station_df[ALL_NUMERIC].values

    return df


# ============================================================
# Spatial imputation
# ============================================================

def spatial_imputation(df, k=4):

    df = df.copy()

    distance_matrix = build_distance_matrix()

    station_to_index = {
        station: i
        for i, station in enumerate(STATIONS)
    }

    lookup = {
        (row.datetime, row.station): idx
        for idx, row in df.iterrows()
    }

    for feature in ALL_NUMERIC:

        missing_indices = df.index[
            df[feature].isna()
        ]

        filled = 0

        for idx in missing_indices:

            timestamp = df.at[
                idx,
                "datetime"
            ]

            station = df.at[
                idx,
                "station"
            ]

            station_idx = station_to_index[
                station
            ]

            distances = (
                distance_matrix[station_idx]
                .copy()
            )

            distances[station_idx] = np.inf

            nearest = np.argsort(
                distances
            )[:k]

            values = []
            weights = []

            for neighbor_idx in nearest:

                neighbor = STATIONS[
                    neighbor_idx
                ]

                neighbor_idx_row = lookup.get(
                    (timestamp, neighbor)
                )

                if neighbor_idx_row is None:
                    continue

                value = df.at[
                    neighbor_idx_row,
                    feature
                ]

                if pd.isna(value):
                    continue

                distance = distances[
                    neighbor_idx
                ]

                weight = 1.0 / max(
                    distance,
                    1e-6
                )

                values.append(value)
                weights.append(weight)

            if values:

                values = np.asarray(values)
                weights = np.asarray(weights)

                df.at[
                    idx,
                    feature
                ] = np.sum(
                    values * weights
                ) / np.sum(weights)

                filled += 1

        print(
            f"{feature}: spatially filled "
            f"{filled:,}"
        )

    return df


# ============================================================
# Median fallback
# ============================================================

def median_fallback(df, train_medians=None):

    df = df.copy()

    if train_medians is None:

        train_medians = (
            df.groupby("station")[ALL_NUMERIC]
            .median()
        )

    for station in STATIONS:

        mask = df["station"] == station

        for feature in ALL_NUMERIC:

            median_value = train_medians.loc[
                station,
                feature
            ]

            df.loc[
                mask & df[feature].isna(),
                feature
            ] = median_value

    return df


# ============================================================
# Preprocess one split
# ============================================================

def preprocess_split(df, train_medians=None):

    df = causal_temporal_imputation(
        df,
        max_gap=SHORT_GAP_LIMIT
    )

    df = spatial_imputation(
        df,
        k=4
    )

    df = median_fallback(
        df,
        train_medians=train_medians
    )

    return df


# ============================================================
# Convert dataframe to [time, station, feature]
# ============================================================

def to_tensor_format(df, features):

    pivoted = (
        df.pivot(
            index="datetime",
            columns="station",
            values=features
        )
    )

    pivoted = pivoted.sort_index()

    # Ensure station order is fixed.
    arrays = []

    for feature in features:

        feature_data = pivoted[
            feature
        ].reindex(
            columns=STATIONS
        )

        arrays.append(
            feature_data.values
        )

    # [features, time, stations]
    array = np.stack(
        arrays,
        axis=-1
    )

    # [time, stations, features]
    array = np.transpose(
        array,
        (0, 1, 2)
    )

    return array.astype(
        np.float32
    )


# ============================================================
# Create sliding windows
# ============================================================

def create_windows(
    pollution,
    weather,
    targets,
    input_length=72,
    horizon=24
):

    X_pollution = []
    X_weather = []
    Y = []

    total_time = len(
        pollution
    )

    for end in range(
        input_length,
        total_time - horizon + 1
    ):

        start = end - input_length

        X_pollution.append(
            pollution[
                start:end
            ]
        )

        X_weather.append(
            weather[
                start:end
            ]
        )

        Y.append(
            targets[
                end:end + horizon
            ]
        )

    return (
        np.asarray(
            X_pollution,
            dtype=np.float32
        ),
        np.asarray(
            X_weather,
            dtype=np.float32
        ),
        np.asarray(
            Y,
            dtype=np.float32
        ),
    )


# ============================================================
# Main
# ============================================================

def main():

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    print("Loading raw data...")

    df = load_data()

    print(
        f"Total rows: {len(df):,}"
    )

    # --------------------------------------------------------
    # Wind encoding
    # --------------------------------------------------------

    print(
        "\nEncoding wind direction..."
    )

    df = encode_wind_direction(df)

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    train, val, test = chronological_split(
        df
    )

    # --------------------------------------------------------
    # Train medians
    # --------------------------------------------------------

    train_medians = (
        train.groupby("station")[ALL_NUMERIC]
        .median()
    )

    # --------------------------------------------------------
    # Imputation
    # --------------------------------------------------------

    print("\nProcessing TRAIN...")
    train = preprocess_split(
        train,
        train_medians
    )

    print("\nProcessing VALIDATION...")
    val = preprocess_split(
        val,
        train_medians
    )

    print("\nProcessing TEST...")
    test = preprocess_split(
        test,
        train_medians
    )

    # --------------------------------------------------------
    # Check missing values
    # --------------------------------------------------------

    print("\nRemaining missing values:")

    print(
        "Train:",
        train[ALL_NUMERIC]
        .isna()
        .sum()
        .sum()
    )

    print(
        "Validation:",
        val[ALL_NUMERIC]
        .isna()
        .sum()
        .sum()
    )

    print(
        "Test:",
        test[ALL_NUMERIC]
        .isna()
        .sum()
        .sum()
    )

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    print(
        "\nFitting scalers using TRAIN only..."
    )

    pollution_scaler = StandardScaler()

    weather_scaler = StandardScaler()

    pollution_scaler.fit(
        train[POLLUTION_FEATURES]
    )

    weather_scaler.fit(
        train[
            [
                "TEMP",
                "PRES",
                "DEWP",
                "RAIN",
                "WSPM",
            ]
        ]
    )

    # Wind sin/cos already lie in [-1, 1].
    # Keep them as-is.

    for split in [train, val, test]:

        split[POLLUTION_FEATURES] = (
            pollution_scaler.transform(
                split[POLLUTION_FEATURES]
            )
        )

        split[
            [
                "TEMP",
                "PRES",
                "DEWP",
                "RAIN",
                "WSPM",
            ]
        ] = weather_scaler.transform(
            split[
                [
                    "TEMP",
                    "PRES",
                    "DEWP",
                    "RAIN",
                    "WSPM",
                ]
            ]
        )

    # --------------------------------------------------------
    # Convert to tensors
    # --------------------------------------------------------

    train_pollution = to_tensor_format(
        train,
        POLLUTION_FEATURES
    )

    train_weather = to_tensor_format(
        train,
        WEATHER_FEATURES
    )

    val_pollution = to_tensor_format(
        val,
        POLLUTION_FEATURES
    )

    val_weather = to_tensor_format(
        val,
        WEATHER_FEATURES
    )

    test_pollution = to_tensor_format(
        test,
        POLLUTION_FEATURES
    )

    test_weather = to_tensor_format(
        test,
        WEATHER_FEATURES
    )

    # Target is PM2.5.
    train_target = train_pollution[:, :, 0]
    val_target = val_pollution[:, :, 0]
    test_target = test_pollution[:, :, 0]

    # --------------------------------------------------------
    # Windows
    # --------------------------------------------------------

    Xp_train, Xw_train, Y_train = create_windows(
        train_pollution,
        train_weather,
        train_target,
        INPUT_LENGTH,
        FORECAST_HORIZON
    )

    Xp_val, Xw_val, Y_val = create_windows(
        val_pollution,
        val_weather,
        val_target,
        INPUT_LENGTH,
        FORECAST_HORIZON
    )

    Xp_test, Xw_test, Y_test = create_windows(
        test_pollution,
        test_weather,
        test_target,
        INPUT_LENGTH,
        FORECAST_HORIZON
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    np.savez_compressed(
        "data/processed/train.npz",
        X_pollution=Xp_train,
        X_weather=Xw_train,
        Y=Y_train
    )

    np.savez_compressed(
        "data/processed/val.npz",
        X_pollution=Xp_val,
        X_weather=Xw_val,
        Y=Y_val
    )

    np.savez_compressed(
        "data/processed/test.npz",
        X_pollution=Xp_test,
        X_weather=Xw_test,
        Y=Y_test
    )

    joblib.dump(
        pollution_scaler,
        "data/processed/pollution_scaler.pkl"
    )

    joblib.dump(
        weather_scaler,
        "data/processed/weather_scaler.pkl"
    )

    print("\n========================================")
    print("FINAL DATASET SHAPES")
    print("========================================")

    print(
        f"Train pollution : {Xp_train.shape}"
    )

    print(
        f"Train weather   : {Xw_train.shape}"
    )

    print(
        f"Train target    : {Y_train.shape}"
    )

    print(
        f"\nVal pollution   : {Xp_val.shape}"
    )

    print(
        f"Val weather     : {Xw_val.shape}"
    )

    print(
        f"Val target      : {Y_val.shape}"
    )

    print(
        f"\nTest pollution  : {Xp_test.shape}"
    )

    print(
        f"Test weather    : {Xw_test.shape}"
    )

    print(
        f"Test target     : {Y_test.shape}"
    )

    print(
        "\nSaved processed datasets to "
        "data/processed/"
    )


if __name__ == "__main__":
    main()