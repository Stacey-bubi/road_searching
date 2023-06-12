import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CustomDataset(Dataset):
    def __init__(self, root_dir, image_size, image_transform=None, mask_transform=None):
        self.root_dir = root_dir
        self.image_size = image_size
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.image_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_filename = self.image_files[idx]
        if not image_filename.endswith('.jpg'):
            return self.__getitem__((idx + 1) % self.__len__())  # skip non-jpg files
        mask_filename = image_filename.replace("_sat.jpg", "_mask.png")

        image_path = os.path.join(self.root_dir, image_filename)
        mask_path = os.path.join(self.root_dir, mask_filename)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.image_size is not None:
            image = image.resize(self.image_size, Image.BILINEAR)
            mask = mask.resize(self.image_size, Image.NEAREST)

        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return image, mask
