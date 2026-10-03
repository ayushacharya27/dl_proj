import numpy as np
import pandas as pd

from data.station_metadata import STATIONS
from data.graph import build_distance_matrix


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
]

NUMERIC_FEATURES = POLLUTION_FEATURES + WEATHER_FEATURES


def load_combined_data():
    """
    Load all station CSV files and combine them into one dataframe.
    """

    from data.preprocessing import load_station_file

    import glob

    files = sorted(glob.glob("data/raw/PRSA_Data_*.csv"))

    dataframes = []

    for file in files:
        df = load_station_file(file)
        dataframes.append(df)

    combined = pd.concat(dataframes, ignore_index=True)

    combined = combined.sort_values(
        ["datetime", "station"]
    ).reset_index(drop=True)

    return combined


def temporal_interpolation(df, max_gap=6):
    """
    Fill short numerical gaps using time-based interpolation.

    Only gaps up to max_gap consecutive missing values are filled.
    """

    df = df.copy()

    for station in STATIONS:

        mask = df["station"] == station

        station_data = (
            df.loc[mask]
            .sort_values("datetime")
            .copy()
        )

        station_data = station_data.set_index("datetime")

        for feature in NUMERIC_FEATURES:

            station_data[feature] = (
                station_data[feature]
                .interpolate(
                    method="time",
                    limit=max_gap,
                    limit_direction="both"
                )
            )

        df.loc[mask, NUMERIC_FEATURES] = (
            station_data[NUMERIC_FEATURES].values
        )

    return df


def spatial_imputation(df, k=4):
    """
    Fill remaining missing values using spatial neighbors.

    For every missing station-feature-timestamp value,
    use available values from nearby stations at the same timestamp.

    Inverse-distance weighting is used.
    """

    df = df.copy()

    distance_matrix = build_distance_matrix()

    station_to_index = {
        station: i
        for i, station in enumerate(STATIONS)
    }

    # Create fast lookup:
    # (datetime, station) -> dataframe index
    lookup = {
        (row.datetime, row.station): idx
        for idx, row in df.iterrows()
    }

    for feature in NUMERIC_FEATURES:

        missing_indices = df.index[
            df[feature].isna()
        ]

        print(
            f"\nSpatial imputation for {feature}: "
            f"{len(missing_indices)} missing values"
        )

        filled = 0

        for idx in missing_indices:

            timestamp = df.at[idx, "datetime"]
            station = df.at[idx, "station"]

            station_idx = station_to_index[station]

            distances = distance_matrix[station_idx].copy()

            distances[station_idx] = np.inf

            nearest_indices = np.argsort(distances)[:k]

            values = []
            weights = []

            for neighbor_idx in nearest_indices:

                neighbor = STATIONS[neighbor_idx]

                neighbor_key = (timestamp, neighbor)

                neighbor_row = lookup.get(neighbor_key)

                if neighbor_row is None:
                    continue

                value = df.at[neighbor_row, feature]

                if pd.isna(value):
                    continue

                distance = distances[neighbor_idx]

                # Avoid division by zero
                weight = 1.0 / max(distance, 1e-6)

                values.append(value)
                weights.append(weight)

            if values:

                weighted_value = (
                    np.sum(
                        np.array(values)
                        * np.array(weights)
                    )
                    /
                    np.sum(weights)
                )

                df.at[idx, feature] = weighted_value
                filled += 1

        print(f"Filled spatially: {filled}")

    return df


def fallback_imputation(df):
    """
    Final fallback for values that remain missing.

    Uses the station-wise median for each feature.
    """

    df = df.copy()

    remaining_before = df[NUMERIC_FEATURES].isna().sum().sum()

    print(
        f"\nRemaining missing values before fallback: "
        f"{remaining_before}"
    )

    for station in STATIONS:

        mask = df["station"] == station

        for feature in NUMERIC_FEATURES:

            median_value = df.loc[
                mask, feature
            ].median()

            if pd.notna(median_value):

                df.loc[
                    mask & df[feature].isna(),
                    feature
                ] = median_value

    remaining_after = df[NUMERIC_FEATURES].isna().sum().sum()

    print(
        f"Remaining missing values after fallback: "
        f"{remaining_after}"
    )

    return df


def report_missing_values(df, title):

    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")

    missing = df[NUMERIC_FEATURES].isna().sum()

    print(missing)

    print(
        f"\nTotal missing values: "
        f"{missing.sum()}"
    )


def main():

    print("Loading combined dataset...")

    df = load_combined_data()

    report_missing_values(
        df,
        "INITIAL MISSING VALUES"
    )

    # --------------------------------------------------
    # STEP 1: Temporal interpolation
    # --------------------------------------------------

    print(
        "\nRunning temporal interpolation "
        "for gaps <= 6 hours..."
    )

    df = temporal_interpolation(
        df,
        max_gap=6
    )

    report_missing_values(
        df,
        "AFTER TEMPORAL INTERPOLATION"
    )

    # --------------------------------------------------
    # STEP 2: Spatial interpolation
    # --------------------------------------------------

    print(
        "\nRunning spatial interpolation..."
    )

    df = spatial_imputation(
        df,
        k=4
    )

    report_missing_values(
        df,
        "AFTER SPATIAL IMPUTATION"
    )

    # --------------------------------------------------
    # STEP 3: Final fallback
    # --------------------------------------------------

    print(
        "\nRunning final station-median fallback..."
    )

    df = fallback_imputation(df)

    report_missing_values(
        df,
        "FINAL DATASET"
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    output_path = (
        "data/processed/combined_imputed.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nSaved processed dataset to:"
        f"\n{output_path}"
    )


if __name__ == "__main__":
    main()