import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class RainMixDataset(Dataset):
    def __init__(self,
                 rain100h_path,
                 rain100l_path,
                 transform=None,
                 size=128,
                 ratio=0.5):  # H 的比例（0.5 = 均衡）
        
        self.rain100h_rain = os.path.join(rain100h_path, "rain")
        self.rain100h_clean = os.path.join(rain100h_path, "clean")

        self.rain100l_rain = os.path.join(rain100l_path, "rain")
        self.rain100l_clean = os.path.join(rain100l_path, "clean")

        self.h_files = sorted(os.listdir(self.rain100h_rain))
        self.l_files = sorted(os.listdir(self.rain100l_rain))

        self.ratio = ratio

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3)
            ])
        else:
            self.transform = transform

        # 为了让 DataLoader 有长度
        self.length = max(len(self.h_files), len(self.l_files))

    def __len__(self):
        return self.length

    def load_pair(self, rain_dir, clean_dir, file_list, idx):
        file_name = file_list[idx % len(file_list)]

        rain_path = os.path.join(rain_dir, file_name)
        clean_path = os.path.join(clean_dir, file_name)

        rain_img = Image.open(rain_path).convert("RGB")
        clean_img = Image.open(clean_path).convert("RGB")

        rain_img = self.transform(rain_img)
        clean_img = self.transform(clean_img)

        return rain_img, clean_img

    def __getitem__(self, idx):
        # 随机选择 H 或 L
        if random.random() < self.ratio:
            rain_img, clean_img = self.load_pair(
                self.rain100h_rain,
                self.rain100h_clean,
                self.h_files,
                idx
            )
            rain_level = 1  # heavy
        else:
            rain_img, clean_img = self.load_pair(
                self.rain100l_rain,
                self.rain100l_clean,
                self.l_files,
                idx
            )
            rain_level = 0  # light

        return {
            "rain": rain_img,
            "clean": clean_img,
            "label": torch.tensor(rain_level)
        }