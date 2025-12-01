import random

import torch
import torchvision.transforms.functional as TF
from torchvision import transforms


class ImageAugmentation:
    def __init__(
        self,
        affine,
        rotation=False,
        pixel_offset=None,
        flip=False,
        noise_level=None,
        image_size=(28, 28),
        zero_channel_idxs=None,
        random_colouring=True,
    ):
        """
        Initialize the data augmentation pipeline.

        Args:
            rotation (bool): Whether to apply rotation.
            pixel_offset (tuple): Pixel offset range as (min_offset, max_offset).
            flip (bool): Whether to apply random flips.
            noise_level (float): Maximum intensity of Gaussian noise (0 to 1).
            zoom_level (tuple): Zoom factor range as (min_zoom, max_zoom).
            image_size (tuple): Final image size after augmentation.
        """
        self.rotation = rotation
        self.pixel_offset = pixel_offset
        self.flip = flip
        self.noise_level = noise_level
        self.image_size = image_size
        self.zero_channel_idxs = zero_channel_idxs
        self.random_colouring = random_colouring
        self.affine = affine

    def random_colour(self, img):
        """Randomly apply colour to the image."""
        img_rgb = img.repeat(3, 1, 1)  # Shape: [3, 28, 28]

        # Generate random RGB values and apply them to each channel
        # Shape: [3, 1, 1] for broadcasting
        # random_rgb_factors = torch.rand(3, 1, 1) # Specify distribution in config

        random_rgb_factors = self.random_colouring.sample((3, 1, 1))

        img_rgb = img_rgb * random_rgb_factors  # Apply random tinting

        random_rgb_factors = random_rgb_factors.squeeze(dim=(-1, -2))

        return img_rgb, random_rgb_factors

    def apply_affine(self, img, affine):
        """Apply affine transformation."""

        affine_values = {}

        for key, transform in affine.items():
            if transform["sampler"] is not None:
                # sample if a dist is specified
                if key == "translate" or key == "shear":
                    affine_values[key] = [
                        float(transform["sampler"].sample()),
                        float(transform["sampler"].sample()),
                    ]
                else:
                    affine_values[key] = float(transform["sampler"].sample())
            else:
                affine_values[key] = transform["value"]

        return transforms.functional.affine(
            img,
            angle=affine_values["angle"],
            translate=tuple(affine_values["translate"]),
            scale=affine_values["scale"],
            shear=tuple(affine_values["shear"]),
        ), affine_values

    def resize(self, img):
        """Resize the image to the target size."""
        return transforms.functional.resize(img, self.image_size)

    def zero_channels(self, img):
        mask = torch.ones_like(img)
        mask[self.zero_channel_idxs] = 0
        return img * mask

    def apply_rotation(self, img, degrees=None):
        """Apply rotation."""
        return transforms.functional.rotate(img, degrees)

    def apply_offset(self, img):
        """Randomly apply offset within the specified range."""
        if self.pixel_offset is not None:
            direction = random.choice(["None", "Up", "Down", "Left", "Right"])
            if direction != "None":
                offset = random.randint(self.pixel_offset[0], self.pixel_offset[1])
                pad = [0, 0, 0, 0]  # Padding: [left, top, right, bottom]
                if direction == "Up":
                    pad[1] = offset
                elif direction == "Down":
                    pad[3] = offset
                elif direction == "Left":
                    pad[0] = offset
                elif direction == "Right":
                    pad[2] = offset

                img = transforms.functional.pad(img, pad, fill=0)
        return img

    def apply_flip(self, img):
        """Randomly apply horizontal or vertical flip."""
        if self.flip:
            flip_type = random.choice(["None", "Horizontal", "Vertical"])
            if flip_type == "Horizontal":
                img = transforms.functional.hflip(img)
            elif flip_type == "Vertical":
                img = transforms.functional.vflip(img)
        return img

    def apply_noise(self, img):
        """Randomly add Gaussian noise."""
        if self.noise_level is not None:
            noise = torch.randn_like(img) * self.noise_level
            img = img + noise
            img = torch.clamp(img, 0, 1)  # Ensure pixel values are valid
        return img

    def apply_zoom(self, img):
        """Randomly apply zoom within the specified range."""
        if self.zoom_level is not None:
            zoom_factor = random.uniform(self.zoom_level[0], self.zoom_level[1])
            img = transforms.functional.affine(
                img, angle=0, translate=[0, 0], scale=zoom_factor, shear=[0, 0]
            )
        return img

    def apply_transformations(self, img):
        if self.affine:
            """
            go through the defined distributions and apply them
            """
            img, affine_values = self.apply_affine(img, self.affine)

        colour_factors = None
        if self.random_colouring:
            img, colour_factors = self.random_colour(img)

        if self.zero_channel_idxs is not None:
            img = self.zero_channels(img)
            colour_factors[self.zero_channel_idxs] = 0

        if self.image_size is not None:
            img = self.resize(img)

        return {
            "img": img,
            "colour_factors": colour_factors,
            "zero_channel_idxs": self.zero_channel_idxs,
            "affine": affine_values
            if self.affine
            else torch.tensor(float("nan"), dtype=torch.float32),
        }


class Augmentation:
    def __init__(
        self,
        affine=None,
        rotation=None,
        pixel_offset=None,
        flip=False,
        noise_level=None,
        image_size=None,
        zoom=None,
    ):
        """
        Initialize the data augmentation pipeline.

        Args:
            rotation (bool): Whether to apply rotation.
            pixel_offset (tuple): Pixel offset range as (min_offset, max_offset).
            flip (bool): Whether to apply random flips.
            noise_level (float): Maximum intensity of Gaussian noise (0 to 1).
            zoom_level (tuple): Zoom factor range as (min_zoom, max_zoom).
            image_size (tuple): Final image size after augmentation.
        """
        self.rotation = rotation
        self.pixel_offset = pixel_offset
        self.flip = flip
        self.noise_level = noise_level
        self.image_size = image_size
        self.affine = affine
        self.zoom = zoom

    def apply_affine(self, X, affine):
        """Apply affine transformation."""

        affine_values = {}

        for key, transform in affine.items():
            if transform["sampler"] is not None:
                # sample if a dist is specified
                if key == "translate" or key == "shear":
                    affine_values[key] = [
                        float(transform["sampler"].sample()),
                        float(transform["sampler"].sample()),
                    ]
                else:
                    affine_values[key] = float(transform["sampler"].sample())
            else:
                affine_values[key] = transform["value"]

        return transforms.functional.affine(
            X,
            angle=affine_values["angle"],
            translate=tuple(affine_values["translate"]),
            scale=affine_values["scale"],
            shear=tuple(affine_values["shear"]),
        ), affine_values

    def resize(self, X):
        """Resize the image to the target size."""
        return transforms.functional.resize(X, self.image_size)

    def zero_channels(self, X):
        mask = torch.ones_like(X)
        mask[self.zero_channel_idxs] = 0
        return X * mask

    def apply_rotation(self, X, degrees=None):
        """Apply rotation."""
        return transforms.functional.rotate(X, degrees)

    def apply_offset(self, X):
        """Randomly apply offset within the specified range."""
        if self.pixel_offset is not None:
            direction = random.choice(["None", "Up", "Down", "Left", "Right"])
            if direction != "None":
                offset = random.randint(self.pixel_offset[0], self.pixel_offset[1])
                pad = [0, 0, 0, 0]  # Padding: [left, top, right, bottom]
                if direction == "Up":
                    pad[1] = offset
                elif direction == "Down":
                    pad[3] = offset
                elif direction == "Left":
                    pad[0] = offset
                elif direction == "Right":
                    pad[2] = offset

                X = transforms.functional.pad(X, pad, fill=0)
        return X

    def apply_flip(self, X):
        """Randomly apply horizontal or vertical flip."""
        if self.flip:
            flip_type = random.choice(["None", "Horizontal", "Vertical"])
            if flip_type == "Horizontal":
                X = transforms.functional.hflip(X)
            elif flip_type == "Vertical":
                X = transforms.functional.vflip(X)
        return X

    def apply_noise(self, X):
        """Randomly add Gaussian noise."""
        if self.noise_level is not None:
            noise = torch.randn_like(X) * self.noise_level
            X = X + noise
            X = torch.clamp(X, 0, 1)  # Ensure pixel values are valid
        return X

    def apply_zoom(self, X):
        """Randomly apply zoom within the specified range."""
        if self.zoom_level is not None:
            zoom_factor = random.uniform(self.zoom_level[0], self.zoom_level[1])
            X = transforms.functional.affine(
                X, angle=0, translate=[0, 0], scale=zoom_factor, shear=[0, 0]
            )
        return X

    def apply_transformations(self, X):
        if self.affine:
            """
            Go through the defined distributions and apply them
            """
            X, affine_values = self.apply_affine(X, self.affine)

        if self.zoom:
            X = self.apply_zoom(X)

        if self.flip:
            X = self.apply_flip(X)

        if self.image_size is not None:
            X = self.resize(X)

        return X


class FlipAndRotate:
    """
    Used for E-MNIST dataset.
    """

    def __call__(self, img):
        return TF.rotate(TF.hflip(img), angle=90)


class Choice:
    """
    Used to select options from a specified list at uniform random.
    """

    def __init__(self, choices: list):
        self.choices = choices

    def sample(self):
        return self.choices[torch.randint(0, len(self.choices), (1,)).item()]
