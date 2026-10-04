import os
import glob
import pickle
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from config import Config

class ABIDEDataset(Dataset):
    """
    ABIDE Dataset for 2D sMRI slices.
    Applies per-slice min-max normalization to [0, 1].
    Optional in-memory caching to eliminate disk I/O bottlenecks.
    """
    def __init__(self, file_paths, labels, cache_in_memory=True):
        self.file_paths = file_paths
        self.labels = labels
        self.cache_in_memory = cache_in_memory
        self.cached_images = {}

        if self.cache_in_memory:
            print(f"[INFO] Preloading {len(self.file_paths)} slices into memory...")
            for i, p in enumerate(self.file_paths):
                img = self._load_and_preprocess(p)
                self.cached_images[i] = img
            print(f"[INFO] Finished preloading {len(self.file_paths)} slices.")

    def _load_and_preprocess(self, path):
        nii = nib.load(path)
        data = nii.get_fdata(dtype=np.float32)

        # Handle 2D slice
        if data.ndim == 3 and data.shape[-1] == 1:
            data = data.squeeze(-1)
        elif data.ndim != 2:
            # If unexpected shape, take central slice
            data = data[:, :, data.shape[-1] // 2]

        # Min-max normalization per slice to [0, 1]
        d_min = data.min()
        d_max = data.max()
        if d_max > d_min:
            data = (data - d_min) / (d_max - d_min)
        else:
            data = np.zeros_like(data, dtype=np.float32)

        # Shape to (1, H, W)
        tensor_data = torch.from_numpy(data).unsqueeze(0).float()
        return tensor_data

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        if self.cache_in_memory and idx in self.cached_images:
            img = self.cached_images[idx]
        else:
            img = self._load_and_preprocess(self.file_paths[idx])

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label

def get_or_create_split(data_dir=Config.DATA_DIR, split_file=Config.SPLIT_FILE, test_size=Config.TEST_SPLIT_RATIO, seed=Config.SEED):
    """
    Scans dataset directory, checks if split_indices.pkl exists;
    if not, creates stratified 80/20 train/test split and saves it.
    """
    if os.path.exists(split_file):
        try:
            with open(split_file, "rb") as f:
                split_data = pickle.load(f)
            if split_data.get("train_files") and os.path.exists(split_data["train_files"][0]):
                print(f"[INFO] Loading existing train/test split from {split_file}")
                return split_data
            else:
                print(f"[WARN] Paths in {split_file} do not exist on this OS. Creating fresh split.")
        except Exception:
            pass

    print("[INFO] Creating fresh stratified train/test split...")
    autistic_dir = os.path.join(data_dir, "Autistic")
    control_dir = os.path.join(data_dir, "Typical_Control")

    autistic_files = sorted(glob.glob(os.path.join(autistic_dir, "*.nii")))
    control_files = sorted(glob.glob(os.path.join(control_dir, "*.nii")))

    print(f"[INFO] Found {len(autistic_files)} Autistic and {len(control_files)} Typical Control slices.")

    all_files = autistic_files + control_files
    # Label: 1 for Autistic, 0 for Typical_Control
    all_labels = [1] * len(autistic_files) + [0] * len(control_files)

    train_files, test_files, train_labels, test_labels = train_test_split(
        all_files,
        all_labels,
        test_size=test_size,
        stratify=all_labels,
        random_state=seed,
        shuffle=True
    )

    split_data = {
        "train_files": train_files,
        "test_files": test_files,
        "train_labels": train_labels,
        "test_labels": test_labels,
        "seed": seed,
        "test_ratio": test_size,
        "train_counts": {
            "Autistic (1)": sum(train_labels),
            "Typical_Control (0)": len(train_labels) - sum(train_labels),
            "Total": len(train_labels)
        },
        "test_counts": {
            "Autistic (1)": sum(test_labels),
            "Typical_Control (0)": len(test_labels) - sum(test_labels),
            "Total": len(test_labels)
        }
    }

    with open(split_file, "wb") as f:
        pickle.dump(split_data, f)
    print(f"[INFO] Stratified train/test split saved to {split_file}")
    return split_data

def get_data_loaders(batch_size=Config.BATCH_SIZE, cache_in_memory=Config.CACHE_IN_MEMORY, num_workers=Config.NUM_WORKERS):
    """
    Returns (train_loader, test_loader, split_data)
    """
    split_data = get_or_create_split()

    train_dataset = ABIDEDataset(
        file_paths=split_data["train_files"],
        labels=split_data["train_labels"],
        cache_in_memory=cache_in_memory
    )
    test_dataset = ABIDEDataset(
        file_paths=split_data["test_files"],
        labels=split_data["test_labels"],
        cache_in_memory=cache_in_memory
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, test_loader, split_data

if __name__ == "__main__":
    split_data = get_or_create_split()
    print("Train set distribution:", split_data["train_counts"])
    print("Test set distribution :", split_data["test_counts"])
    print("Testing data loader instantiation...")
    train_loader, test_loader, _ = get_data_loaders(batch_size=8, cache_in_memory=False)
    for x, y in train_loader:
        print(f"Batch X shape: {x.shape}, Batch Y shape: {y.shape}, Range: [{x.min():.3f}, {x.max():.3f}]")
        break
    print("Data loader test passed successfully!")
