import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class AirQualityDataset(Dataset):

    def __init__(self, npz_path):

        data = np.load(npz_path)

        self.X_pollution = torch.tensor(
            data["X_pollution"],
            dtype=torch.float32
        )

        self.X_weather = torch.tensor(
            data["X_weather"],
            dtype=torch.float32
        )

        self.Y = torch.tensor(
            data["Y"],
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):

        return (
            self.X_pollution[idx],
            self.X_weather[idx],
            self.Y[idx]
        )


def create_dataloaders(
    batch_size=32,
    num_workers=2
):

    train_dataset = AirQualityDataset(
        "data/processed/train.npz"
    )

    val_dataset = AirQualityDataset(
        "data/processed/val.npz"
    )

    test_dataset = AirQualityDataset(
        "data/processed/test.npz"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )


if __name__ == "__main__":

    train_loader, val_loader, test_loader = (
        create_dataloaders()
    )

    Xp, Xw, Y = next(iter(train_loader))

    print("Pollution batch:", Xp.shape)
    print("Weather batch:  ", Xw.shape)
    print("Target batch:   ", Y.shape)

    print("\nDataset sizes:")
    print("Train:", len(train_loader.dataset))
    print("Val:  ", len(val_loader.dataset))
    print("Test: ", len(test_loader.dataset))