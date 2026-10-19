"""Manifest-driven dataset. Authentic images carry an all-zero mask, which is a trusted mask."""
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ForgeDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, split, augment: bool = False, include_removal_train: bool = False):
        df = manifest[manifest["split"].isin([split] if isinstance(split, str) else split)]
        self.df = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cv2.cvtColor(cv2.imread(r["path"]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(r["mask_path"], cv2.IMREAD_GRAYSCALE) if isinstance(r["mask_path"], str) and r["mask_path"] \
            else np.zeros(img.shape[:2], np.uint8)
        mask = (mask > 127).astype(np.float32)
        if self.augment:  # flips only: resizing/blur would erase the traces the forensic branch relies on
            if np.random.rand() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            if np.random.rand() < 0.5:
                img, mask = img[::-1], mask[::-1]
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        return {"x": x, "y": int(r["label"]), "mask": torch.from_numpy(np.ascontiguousarray(mask))[None],
                "has_mask": 1, "idx": i}
