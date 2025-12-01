import os

import lightning as L
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms 

from wwdc.data.augmentations import FlipAndRotate

class RGBMNIST(Dataset):
    def __init__(self, y_catalog, train, x_ds):
        """
        Initialize the RGB MNIST dataset.

        Args:
            root (str, optional):
                The path where the MNIST data is to be saved or loaded from.
                Defaults to ".".
            train (bool, optional):
                True for training set, False for test set. Defaults to True.
            image_size: (tuple, optional)
                Rescaling size for image
        """

        self.x_ds = x_ds
        self.train = train
        self.y_catalog = y_catalog

        # Check if data already exists before downloading
        if not self.data_exists:
            print(f"Data not found locally. Downloading to {self.x_ds['fp']}...")

        transforms_ = transforms.Compose([transforms.ToTensor()])

        # Load the MNIST dataset
        try:
            self.mnist_data = datasets.MNIST(
                root=self.x_ds["fp"],
                train=self.train,
                download=not self.data_exists,
                transform=transforms_,
            )
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise e

    @property
    def data_exists(self):
        # MNIST typically has processed folders with train and test files
        processed_dir = os.path.join(self.x_ds["fp"], "MNIST", "raw")
        return os.path.exists(processed_dir)

    def __len__(self):
        return len(self.mnist_data)

    def __getitem__(self, idx):
        # Load a grayscale MNIST image
        img, label = self.mnist_data[idx]

        augmented_data = self.x_ds["augmentations"].apply_transformations(img)
        img_rgb = augmented_data["img"]
        random_rgb_factors = augmented_data["colour_factors"]
        # affine = augmented_data["affine"]

        return {
            "X": img_rgb,
            "catalog": {
                "r": random_rgb_factors[0].unsqueeze(0),
                "g": random_rgb_factors[1].unsqueeze(0),
                "b": random_rgb_factors[2].unsqueeze(0),
                "digit": torch.tensor(label),
                # "angle": torch.tensor(affine["angle"]).unsqueeze(0),
                # "scale": torch.tensor(affine["scale"]).unsqueeze(0),
            },
        }
