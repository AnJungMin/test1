import os
import json
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# 모델 임포트 (상대 경로 조정)
from app.model.your_model import MultiTaskMobileNetV3

# --------------------
# 1) 데이터셋 정의
# --------------------
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

        # 6개 질환 레이블(각 0~3)
        labels = [int(sample[f"value_{i+1}"]) for i in range(6)]
        labels = torch.tensor(labels, dtype=torch.long)

        return image, labels

def main():
    # --------------------
    # 2) 하이퍼파라미터 & 경로 설정
    # --------------------
    json_dir = r"JSON_파일_폴더경로"
    img_dir  = r"이미지_폴더경로"
    num_epochs = 1
    batch_size = 64
    learning_rate = 0.001
    save_path = r"./app/model/MTL_BASIS.pth"

    # --------------------
    # 3) 전처리 정의
    # --------------------
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # --------------------
    # 4) 데이터셋 및 데이터로더
    # --------------------
    dataset = MultiLabelDataset(json_dir, img_dir, transform=data_transforms)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # --------------------
    # 5) 모델 준비
    # --------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskMobileNetV3().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --------------------
    # 6) 학습 루프
    # --------------------
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        task_correct = [0]*6
        total_samples = 0
        overall_correct = 0

        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # 6개 태스크 예측
            mise_head, pizi_head, mosa_head, mono_head, biddem_head, talmo_head = model(images)

            # 6개 태스크 각각 CrossEntropyLoss
            loss_mise   = criterion(mise_head,   labels[:, 0])
            loss_pizi   = criterion(pizi_head,   labels[:, 1])
            loss_mosa   = criterion(mosa_head,   labels[:, 2])
            loss_mono   = criterion(mono_head,   labels[:, 3])
            loss_biddem = criterion(biddem_head, labels[:, 4])
            loss_talmo  = criterion(talmo_head,  labels[:, 5])

            # 총 손실
            loss = loss_mise + loss_pizi + loss_mosa + loss_mono + loss_biddem + loss_talmo
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            # 정확도 계산
            preds = torch.stack([
                mise_head.argmax(dim=1),
                pizi_head.argmax(dim=1),
                mosa_head.argmax(dim=1),
                mono_head.argmax(dim=1),
                biddem_head.argmax(dim=1),
                talmo_head.argmax(dim=1)
            ], dim=1)

            for task in range(6):
                correct_count = (preds[:, task] == labels[:, task]).sum().item()
                task_correct[task] += correct_count
                overall_correct += correct_count

        epoch_loss = running_loss / len(train_loader.dataset)
        overall_accuracy = overall_correct / (total_samples * 6)

        print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}, Overall Acc: {overall_accuracy*100:.2f}%")

    # --------------------
    # 7) 모델 저장
    # --------------------
    torch.save(model, save_path)
    print(f"Model saved to: {save_path}")

    # --------------------
    # 8) 테스트 (옵션)
    # --------------------
    model.eval()
    test_loss = 0.0
    task_correct = [0]*6
    overall_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            mise_head, pizi_head, mosa_head, mono_head, biddem_head, talmo_head = model(images)

            loss = (criterion(mise_head,   labels[:, 0]) +
                    criterion(pizi_head,   labels[:, 1]) +
                    criterion(mosa_head,   labels[:, 2]) +
                    criterion(mono_head,   labels[:, 3]) +
                    criterion(biddem_head, labels[:, 4]) +
                    criterion(talmo_head,  labels[:, 5]))

            test_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            preds = torch.stack([
                mise_head.argmax(dim=1),
                pizi_head.argmax(dim=1),
                mosa_head.argmax(dim=1),
                mono_head.argmax(dim=1),
                biddem_head.argmax(dim=1),
                talmo_head.argmax(dim=1)
            ], dim=1)

            for task in range(6):
                correct_count = (preds[:, task] == labels[:, task]).sum().item()
                task_correct[task] += correct_count
                overall_correct += correct_count

    avg_test_loss = test_loss / total_samples
    overall_accuracy = overall_correct / (total_samples * 6)
    print(f"Test Loss: {avg_test_loss:.4f}, Test Overall Acc: {overall_accuracy*100:.2f}%")

if __name__ == "__main__":
    main()
