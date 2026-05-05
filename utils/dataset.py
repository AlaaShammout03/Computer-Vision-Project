import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CrowdCountingDataset(Dataset):
    def __init__(self, csv_path, image_dir, density_dir,
                 crop_size=(384, 512), output_downscale=8, augment=False,
                 density_scale=100.0):
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.density_dir = density_dir
        self.crop_size = crop_size
        self.output_downscale = output_downscale
        self.augment = augment
        self.density_scale = density_scale
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def _resize_density_preserve_count(self, density, new_h, new_w):
        original = density.sum()
        resized = cv2.resize(density, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if resized.sum() > 0 and original > 0:
            resized = resized * (original / resized.sum())
        return resized.astype(np.float32)

    def _random_crop(self, image_np, density):
        H, W = image_np.shape[:2]
        crop_h, crop_w = self.crop_size
        if H < crop_h or W < crop_w:
            scale = max(crop_h / H, crop_w / W) * 1.05
            new_H, new_W = int(H * scale), int(W * scale)
            image_np = cv2.resize(image_np, (new_W, new_H))
            density = self._resize_density_preserve_count(density, new_H, new_W)
            H, W = new_H, new_W
        top = np.random.randint(0, H - crop_h + 1)
        left = np.random.randint(0, W - crop_w + 1)
        return (image_np[top:top + crop_h, left:left + crop_w],
                density[top:top + crop_h, left:left + crop_w])

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_name = row["image"]
        actual_count = float(row["actual_count"])
        image = Image.open(os.path.join(self.image_dir, image_name)).convert("RGB")
        density = np.load(os.path.join(self.density_dir, row["density_map"])).astype(np.float32)
        image_np = np.array(image)

        if self.augment:
            image_np, density = self._random_crop(image_np, density)
            if np.random.rand() < 0.5:
                image_np = np.ascontiguousarray(image_np[:, ::-1, :])
                density = np.ascontiguousarray(density[:, ::-1])
            image_pil = Image.fromarray(image_np)
            if np.random.rand() < 0.5:
                image_pil = ImageEnhance.Brightness(image_pil).enhance(np.random.uniform(0.8, 1.2))
            if np.random.rand() < 0.5:
                image_pil = ImageEnhance.Contrast(image_pil).enhance(np.random.uniform(0.8, 1.2))
            image_np = np.array(image_pil)
        else:
            crop_h, crop_w = self.crop_size
            image_np = cv2.resize(image_np, (crop_w, crop_h))
            density = self._resize_density_preserve_count(density, crop_h, crop_w)

        out_h = self.crop_size[0] // self.output_downscale
        out_w = self.crop_size[1] // self.output_downscale
        density = self._resize_density_preserve_count(density, out_h, out_w)

        density = density * self.density_scale

        image_tensor = self.normalize(self.to_tensor(Image.fromarray(image_np)))
        density_tensor = torch.tensor(density, dtype=torch.float32).unsqueeze(0)
        count_tensor = torch.tensor(actual_count, dtype=torch.float32)
        return image_tensor, density_tensor, count_tensor, image_name
