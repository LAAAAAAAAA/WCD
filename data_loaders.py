from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch

__all__ = ["get_loaders"]

# Define image preprocessing transformation
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGBA")  # Ensure the image is in 4-channel RGBA format
        label = self.labels[idx]

        # Resize the image to 256x256
        image = image.resize((256, 256))

        # Convert the image to a numpy array
        image_array = np.array(image)

        # Convert the numpy array to a PyTorch tensor
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0  # Convert shape from (H, W, C) to (C, H, W) and normalize

        # Apply data augmentation transformations (if any)
        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, label

def get_loaders(data_root: str, batch_size: int = 32, num_workers: int = 4):
    root = Path(data_root)

    # Get image paths and labels for training, validation, and test sets
    def get_image_paths_and_labels(folder_path):
        image_paths = []
        labels = []
        for label_idx, class_folder in enumerate(folder_path.iterdir()):
            for image_path in class_folder.iterdir():
                image_paths.append(str(image_path))
                labels.append(label_idx)
        return image_paths, labels

    train_image_paths, train_labels = get_image_paths_and_labels(root / "train")
    val_image_paths, val_labels = get_image_paths_and_labels(root / "val")
    test_image_paths, test_labels = get_image_paths_and_labels(root / "test")

    # Create datasets
    train_dataset = CustomImageDataset(train_image_paths, train_labels)
    val_dataset = CustomImageDataset(val_image_paths, val_labels)
    test_dataset = CustomImageDataset(test_image_paths, test_labels)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


