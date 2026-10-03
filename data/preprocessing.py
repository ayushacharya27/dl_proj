from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

RAW_DIR = Path("data/raw")

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


# ---------------------------------------------------------
# Load one station
# ---------------------------------------------------------

def load_station_file(filepath):
    df = pd.read_csv(filepath)

    # Construct timestamp
    df["datetime"] = pd.to_datetime(
        df[["year", "month", "day", "hour"]]
    )

    # Sort chronologically
    df = df.sort_values("datetime").reset_index(drop=True)

    # Keep only useful columns
    columns = (
        ["datetime"]
        + POLLUTION_FEATURES
        + WEATHER_FEATURES
        + ["wd", "station"]
    )

    df = df[columns]

    return df


# ---------------------------------------------------------
# Load all 12 stations
# ---------------------------------------------------------

def load_all_stations(raw_dir=RAW_DIR):

    files = sorted(raw_dir.glob("PRSA_Data_*.csv"))

    if len(files) == 0:
        raise FileNotFoundError(
            f"No PRSA_Data CSV files found in {raw_dir}"
        )

    print(f"Found {len(files)} station files.")

    stations = []

    for filepath in files:

        print(f"Loading: {filepath.name}")

        df = load_station_file(filepath)

        stations.append(df)

    return stations


# ---------------------------------------------------------
# Check station alignment
# ---------------------------------------------------------

def check_alignment(stations):

    reference_time = stations[0]["datetime"]

    print("\nChecking timestamp alignment...")

    for df in stations:

        station_name = df["station"].iloc[0]

        same_length = len(df) == len(reference_time)
        same_times = df["datetime"].equals(reference_time)

        print(
            f"{station_name:20s} "
            f"rows={len(df):6d} "
            f"aligned={same_length and same_times}"
        )


# ---------------------------------------------------------
# Combine stations
# ---------------------------------------------------------

def combine_stations(stations):

    combined = pd.concat(
        stations,
        ignore_index=True
    )

    combined = combined.sort_values(
        ["datetime", "station"]
    ).reset_index(drop=True)

    return combined


# ---------------------------------------------------------
# Missing-value report
# ---------------------------------------------------------

def missing_value_report(df):

    print("\nMissing values:")
    print("-" * 50)

    missing = df.isna().sum()

    missing = missing[missing > 0]

    if len(missing) == 0:
        print("No missing values.")
        return

    for column, count in missing.items():

        percentage = (
            count / len(df)
        ) * 100

        print(
            f"{column:10s}: "
            f"{count:8d} "
            f"({percentage:.2f}%)"
        )


def station_missing_value_report(stations):

    print("\nMissing values by station:")
    print("=" * 80)

    for df in stations:

        station = df["station"].iloc[0]

        print(f"\n{station}")
        print("-" * 40)

        features = POLLUTION_FEATURES + WEATHER_FEATURES + ["wd"]

        for feature in features:

            missing = df[feature].isna().sum()

            percentage = (
                missing / len(df)
            ) * 100

            print(
                f"{feature:8s}: "
                f"{missing:6d} "
                f"({percentage:5.2f}%)"
            )

def missing_run_analysis(stations):

    features = POLLUTION_FEATURES + WEATHER_FEATURES

    print("\nMissing-value run analysis")
    print("=" * 90)

    for df in stations:

        station = df["station"].iloc[0]

        print(f"\n{station}")
        print("-" * 90)

        for feature in features:

            missing = df[feature].isna()

            # Identify consecutive missing groups
            groups = missing.ne(missing.shift()).cumsum()

            run_lengths = (
                missing.groupby(groups)
                .sum()
            )

            run_lengths = run_lengths[
                run_lengths > 0
            ]

            if len(run_lengths) == 0:
                continue

            max_run = int(run_lengths.max())

            runs_1 = int((run_lengths == 1).sum())
            runs_2_6 = int(
                ((run_lengths >= 2) & (run_lengths <= 6)).sum()
            )
            runs_7_plus = int(
                (run_lengths >= 7).sum()
            )

            print(
                f"{feature:8s} | "
                f"max={max_run:4d}h | "
                f"1h={runs_1:4d} | "
                f"2-6h={runs_2_6:4d} | "
                f"7+h={runs_7_plus:4d}"
            )

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    stations = load_all_stations()

    print("\nNumber of stations:", len(stations))

    check_alignment(stations)

    combined = combine_stations(stations)

    print("\nCombined dataset:")
    print(combined.shape)

    print("\nFirst 5 rows:")
    print(combined.head())

    print("\nStations:")
    print(combined["station"].unique())

    missing_value_report(combined)

    station_missing_value_report(stations)
    missing_run_analysis(stations)