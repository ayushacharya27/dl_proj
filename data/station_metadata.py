import numpy as np
import pandas as pd


# Coordinates are latitude / longitude.
# Keep station order fixed throughout the project.

STATION_COORDINATES = {
    "Aotizhongxin": (39.982, 116.397),
    "Changping": (40.217, 116.230),
    "Dingling": (40.292, 116.220),
    "Dongsi": (39.929, 116.417),
    "Guanyuan": (39.929, 116.339),
    "Gucheng": (39.914, 116.184),
    "Huairou": (40.328, 116.628),
    "Nongzhanguan": (39.937, 116.461),
    "Shunyi": (40.127, 116.655),
    "Tiantan": (39.886, 116.407),
    "Wanliu": (39.987, 116.287),
    "Wanshouxigong": (39.878, 116.352),
}


STATIONS = list(STATION_COORDINATES.keys())


def get_coordinates():

    coordinates = np.array(
        [STATION_COORDINATES[s] for s in STATIONS],
        dtype=np.float32
    )

    return coordinates


def create_station_table():

    coordinates = get_coordinates()

    df = pd.DataFrame(
        coordinates,
        columns=["latitude", "longitude"]
    )

    df.insert(0, "station", STATIONS)

    return df


if __name__ == "__main__":

    df = create_station_table()

    print(df)