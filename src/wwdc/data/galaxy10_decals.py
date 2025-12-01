import os

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from dotenv import load_dotenv
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from torchvision import transforms

load_dotenv()
DATA_DIR = os.getenv("DATA_ROOT") + "/galaxy10_decals"


class Galaxy10(Dataset):
    def __init__(self, x_ds, y_catalog, split="train"):
        if not self.data_exists():
            msg = f"Data not found in {DATA_DIR}. Downloading from HuggingFace..."
            print(msg)
            dataset = load_dataset(
                "mwalmsley/galaxy10_decals",
                "skyviewer",
                cache_dir=DATA_DIR,
            )

            dataset_valtest = dataset["test"].train_test_split(
                test_size=0.5, seed=42, stratify_by_column="label"
            )

            dataset = DatasetDict(
                {
                    "train": dataset["train"],
                    "val": dataset_valtest["train"],
                    "test": dataset_valtest["test"],
                }
            )

            dataset.save_to_disk(DATA_DIR)

        if split not in ["train", "test", "val"]:
            msg = f"Invalid split: {split}. Must be 'train' or 'test' or 'val'."
            raise ValueError(msg)

        self.x_ds = x_ds
        self.y_catalog = y_catalog
        self.dataset = load_from_disk(DATA_DIR)[split]

        self.transform_ = transforms.Compose([transforms.ToTensor()])

        self.labels = {
            0: "disturbed",
            1: "merging",
            2: "round_smooth",
            3: "in_between_round_smooth",
            4: "barred_spiral",
            5: "barred_tight_spiral",
            6: "unbarred_tight_spiral",
            7: "unbarred_loose_spiral",
            8: "edge_on_without_bulge",
            9: "edge_on_with_bulge",
        }

    @staticmethod
    def data_exists():
        # simple check of the path.
        return os.path.exists(DATA_DIR)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # image augmentations here
        item = self.dataset[idx]

        X = self.x_ds["augmentations"].apply_transformations(
            self.transform_(item["image"])
        )

        one_hot_label = one_hot(
            torch.tensor(item["label"]), num_classes=len(self.labels)
        )

        categories = {
            self.labels[idx]: label.unsqueeze(0)
            for idx, label in enumerate(one_hot_label)
        }

        return {"X": X, "catalog": {**categories}, "label": item["label"]}


if __name__ == "__main__":
    data = Galaxy10(x_ds={}, y_catalog={}, split="train")
    # dataset = WWDCDataset(data, return_catalog=True)
    print(data)
    print(len(data))
    print(data[0])
    print(data[0]["X"].shape)
    print(data[0]["X"].max())
    print(data[0]["X"].min())
    print(data[0]["catalog"].keys())
