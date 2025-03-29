import os
import json
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms  # transforms import 추가

class MultiLabelDataset(Dataset):
    def __init__(self, json_dir, img_dir, transform=None):
        self.data = []
        json_files = glob.glob(os.path.join(json_dir, "*.json"))
        
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {json_dir}")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data_part = json.load(f)
                    if isinstance(data_part, dict):
                        self.data.append(data_part)
                    else:
                        self.data.extend(data_part)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue
        
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img_path = os.path.join(self.img_dir, sample["image_file_name"])
        
        # 이미지가 경로에 존재하는지 확인
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")

        # 이미지 전처리가 필요하면 적용
        if self.transform:
            image = self.transform(image)

        # 6개 질환 레이블 (각각 0~3)
        labels = [int(sample[f"value_{i+1}"]) for i in range(6)]
        labels = torch.tensor(labels, dtype=torch.long)

        return image, labels

# 데이터셋을 사용할 때는 transform 인자를 사용하도록 설정
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 예시로 데이터셋을 생성하는 코드
# dataset = MultiLabelDataset(json_dir="your_json_folder", img_dir="your_image_folder", transform=data_transforms)
