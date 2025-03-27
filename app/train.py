import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from app.model.model import MultiTaskMobileNetV3  # 모델 임포트
from app.model.dataset import MultiLabelDataset  # 데이터셋 클래스 임포트
from tqdm import tqdm

# 하이퍼파라미터 설정
num_epochs = 1
learning_rate = 0.001
batch_size = 64

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

json_dir = r"경로\to\json_dir"
img_dir = r"경로\to\image_dir"
dataset = MultiLabelDataset(json_dir, img_dir, transform=data_transforms)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTaskMobileNetV3().to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# 학습 루프
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    task_correct = [0] * 6
    total_samples = 0
    overall_correct = 0

    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = sum([criterion(outputs[i], labels[:, i]) for i in range(6)])
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        preds = torch.stack([output.argmax(dim=1) for output in outputs], dim=1)
        for task in range(6):
            task_correct[task] += (preds[:, task] == labels[:, task]).sum().item()
            overall_correct += (preds[:, task] == labels[:, task]).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    overall_accuracy = overall_correct / (total_samples * 6)

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Overall Accuracy: {overall_accuracy * 100:.2f}%")

torch.save(model, 'app/model/MTL_BASIS.pth')
