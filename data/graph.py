import numpy as np

from data.station_metadata import (
    STATIONS,
    get_coordinates
)


def haversine_distance(lat1, lon1, lat2, lon2):

    R = 6371.0  # Earth radius in km

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def build_distance_matrix():

    coordinates = get_coordinates()

    n = len(coordinates)

    distance_matrix = np.zeros((n, n))

    for i in range(n):

        for j in range(n):

            if i == j:
                continue

            lat1, lon1 = coordinates[i]
            lat2, lon2 = coordinates[j]

            distance_matrix[i, j] = haversine_distance(
                lat1,
                lon1,
                lat2,
                lon2
            )

    return distance_matrix


def build_knn_graph(k=4):

    distance_matrix = build_distance_matrix()

    n = len(STATIONS)

    edges = []

    for i in range(n):

        distances = distance_matrix[i].copy()

        # Don't select the station itself
        distances[i] = np.inf

        nearest = np.argsort(distances)[:k]

        for j in nearest:

            edges.append((i, j))

    return np.array(edges), distance_matrix


if __name__ == "__main__":

    edges, distance_matrix = build_knn_graph(k=4)

    print("\nStations:")
    for i, station in enumerate(STATIONS):
        print(i, station)

    print("\nDistance matrix (km):")
    print(np.round(distance_matrix, 2))

    print("\nKNN edges:")
    print(edges)

    print("\nNumber of directed edges:")
    print(len(edges))