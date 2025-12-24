# CASIA_loader.py
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from config import CASIA_ALIGNED_DATA_DIR

class CASIA(Dataset):
    def __init__(self, root=CASIA_ALIGNED_DATA_DIR):
        self.root = root
        txt_path = os.path.join(root, 'casia-webface.txt')  # keep the same txt file
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"List file not found: {txt_path}")

        image_list, label_list = [], []
        unique_labels = set()

        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                unique_labels.add(int(parts[0]))

        label_map = {l: i for i, l in enumerate(sorted(unique_labels))}
        self.class_nums = len(label_map)

        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                label = label_map[int(parts[0])]
                # Change path to point inside the aligned folder
                img_path = os.path.join(root, parts[1])
                if os.path.exists(img_path):
                    image_list.append(img_path)
                    label_list.append(label)

        self.image_list = image_list
        self.label_list = label_list

        # Only data augmentation + normalization (no alignment needed)
        self.transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[127.5/255] * 3, std=[128/255] * 3)
        ])

        print(f"Loaded {len(self.image_list)} pre-aligned images from {self.class_nums} identities.")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_path = self.image_list[index]
        label = self.label_list[index]

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        return img, label