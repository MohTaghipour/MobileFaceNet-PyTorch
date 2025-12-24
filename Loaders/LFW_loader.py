# LFW_loader.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from config import LFW_ALIGNED_DATA_DIR

class LFW(Dataset):
    def __init__(self, left_paths, right_paths, input_size=112):
        self.left_paths = left_paths
        self.right_paths = right_paths
        self.input_size = input_size

        # Only normalization (no augmentation, no alignment)
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[127.5/255] * 3, std=[128/255] * 3)
        ])

    def __len__(self):
        return len(self.left_paths)

    def _process(self, path):
        img = Image.open(path).convert('RGB')
        return self.transform(img)

    def __getitem__(self, index):
        img_l = self._process(self.left_paths[index])
        img_r = self._process(self.right_paths[index])
        return torch.stack([img_l, img_r])