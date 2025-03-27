import os
import json
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image

class MultiLabelDataset(Dataset):
    def __init__(self, json_dir, img_dir, transform=None):
        self.data = []
        json_files = glob.glob(os.path.join(json_dir, "*.json"))
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data_part = json.load(f)
                # 단일 dict이면 리스트로 감싸주기
                if isinstance(data_part, dict):
                    self.data.append(data_part)
                else:
                    self.data.extend(data_part)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img_path = os.path.join(self.img_dir, sample["image_file_name"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # 6개 질환 레이블 (각각 0~3)
        labels = [int(sample[f"value_{i+1}"]) for i in range(6)]
        labels = torch.tensor(labels, dtype=torch.long)

        return image, labels
