import os

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


def get_transforms(image_size: int = 224, mode: str = "train"):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if mode == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def build_imagefolder(split_dir: str, image_size: int = 224, mode: str = "train"):
    return ImageFolder(root=split_dir, transform=get_transforms(image_size, mode=mode))


def create_deterministic_train_dataset(data_dir, image_size=224):
    return build_imagefolder(
        split_dir=os.path.join(data_dir, "train"),
        image_size=image_size,
        mode="eval",
    )


def create_dataloaders(data_dir, batch_size=84, num_workers=4, image_size=224):
    train_dataset = build_imagefolder(
        split_dir=os.path.join(data_dir, "train"),
        image_size=image_size,
        mode="train",
    )
    valid_dataset = build_imagefolder(
        split_dir=os.path.join(data_dir, "valid"),
        image_size=image_size,
        mode="eval",
    )
    test_dataset = build_imagefolder(
        split_dir=os.path.join(data_dir, "test"),
        image_size=image_size,
        mode="eval",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {
        "train": train_loader,
        "valid": valid_loader,
        "test": test_loader,
        "train_dataset": train_dataset,
        "valid_dataset": valid_dataset,
        "test_dataset": test_dataset,
    }


if __name__ == "__main__":
    data_dir = "dataset_classified_biased"

    if os.path.exists(data_dir):
        loaders = create_dataloaders(data_dir, batch_size=8, num_workers=0)
        deterministic_train_dataset = create_deterministic_train_dataset(data_dir)

        images, labels = next(iter(loaders["train"]))
        print(f"Train batch: images {images.shape}, labels {labels.shape}")
        print(f"Classes: {loaders['train_dataset'].classes}")
        print(f"Deterministic train dataset size: {len(deterministic_train_dataset)}")
    else:
        print(f"Dataset not found at {data_dir}")
        print("Please ensure the dataset is in the correct location.")
