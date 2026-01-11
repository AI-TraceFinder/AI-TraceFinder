from torch.utils.data import DataLoader, Subset
from dataset import TraceFinderDataset
import random


def get_dataloaders(dataset_root, batch_size=64, val_split=0.2, seed=42):
    dataset = TraceFinderDataset(dataset_root, patches_per_image=12)

    size = len(dataset)
    indices = list(range(size))
    random.seed(seed)
    random.shuffle(indices)

    val_size = int(size * val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return train_loader, val_loader













