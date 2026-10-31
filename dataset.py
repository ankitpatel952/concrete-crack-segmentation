# dataset loading for concrete crack images
# main thing here is the patch extraction — images are high-res so we crop 512x512 patches
# during training we bias towards crack-containing patches 50% of the time,
# otherwise you end up with too many blank patches and the model underfits on cracks

import os
from glob import glob
from typing import List, Tuple, Optional
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class ConcreteCrackDataset(Dataset):
    def __init__(self,
                 images_dir: str,
                 masks_dir: str,
                 patch_size: int = 512,
                 is_train: bool = True,
                 file_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg")):
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.patch_size = patch_size
        self.is_train = is_train

        self.image_files = []
        for ext in file_extensions:
            self.image_files.extend(glob(os.path.join(images_dir, f"*{ext}")))
            self.image_files.extend(glob(os.path.join(images_dir, f"*{ext.upper()}")))
        self.image_files.sort()

        if len(self.image_files) == 0:
            raise FileNotFoundError(f"No image files found in {images_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def _apply_augmentations(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        if np.random.rand() > 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)

        k = np.random.choice([0, 1, 2, 3])
        if k > 0:
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()

        # CLAHE helps a lot with concrete — lighting varies a ton between inspection images
        if np.random.rand() > 0.5:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            image = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)
            beta = np.random.uniform(-15, 15)
            image = np.clip(alpha * image.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        return image, mask

    def _extract_patch(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w, _ = image.shape
        ph, pw = self.patch_size, self.patch_size

        if h < ph or w < pw:
            pad_h = max(0, ph - h)
            pad_w = max(0, pw - w)
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            h, w, _ = image.shape

        if self.is_train:
            # half the time, center the patch on an actual crack pixel
            crack_y, crack_x = np.where(mask > 127)
            if len(crack_y) > 0 and np.random.rand() > 0.5:
                idx = np.random.randint(len(crack_y))
                center_y, center_x = crack_y[idx], crack_x[idx]
                y1 = max(0, min(center_y - ph // 2, h - ph))
                x1 = max(0, min(center_x - pw // 2, w - pw))
            else:
                y1 = np.random.randint(0, h - ph + 1)
                x1 = np.random.randint(0, w - pw + 1)
        else:
            # center crop for eval — deterministic
            y1 = (h - ph) // 2
            x1 = (w - pw) // 2

        image_patch = image[y1:y1 + ph, x1:x1 + pw]
        mask_patch = mask[y1:y1 + ph, x1:x1 + pw]
        return image_patch, mask_patch

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_files[idx]
        filename = os.path.basename(img_path)
        base_name, _ = os.path.splitext(filename)

        # try a few common naming patterns for the mask file
        mask_candidates = [
            os.path.join(self.masks_dir, filename),
            os.path.join(self.masks_dir, f"{base_name}.png"),
            os.path.join(self.masks_dir, f"{base_name}.jpg"),
            os.path.join(self.masks_dir, f"{base_name}_mask.png")
        ]
        mask_path = None
        for candidate in mask_candidates:
            if os.path.exists(candidate):
                mask_path = candidate
                break

        if mask_path is None:
            raise FileNotFoundError(f"Mask file not found for {img_path}")

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image_patch, mask_patch = self._extract_patch(image, mask)

        if self.is_train:
            image_patch, mask_patch = self._apply_augmentations(image_patch, mask_patch)

        # standard ImageNet normalization since the encoder is ResNet34 pretrained on ImageNet
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm_img = ((image_patch.astype(np.float32) / 255.0) - mean) / std

        norm_mask = (mask_patch > 127).astype(np.float32)

        img_tensor = torch.from_numpy(norm_img.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(norm_mask).unsqueeze(0).float()

        return img_tensor, mask_tensor
