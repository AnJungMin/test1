import torch
import torch.nn as nn
from torchvision import models

class MultiTaskMobileNetV3(nn.Module):
    def __init__(self):
        super(MultiTaskMobileNetV3, self).__init__()
        # 사전학습된 MobileNetV3-Large 백본 불러오기
        self.backbone = models.mobilenet_v3_large(pretrained=True)
        num_ftrs = self.backbone.classifier[0].in_features
        
        # 마지막 classifier 제거 -> feature vector 추출
        self.backbone.classifier = nn.Identity()
       
        # 각 태스크별 classifier 정의
        self.backbone.fc1 = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs, 4)
        )
        self.backbone.fc2 = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs, 4)
        )
        self.backbone.fc3 = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs, 4)
        )
        self.backbone.fc4 = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs, 4)
        )
        self.backbone.fc5 = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs, 4)
        )
        self.backbone.fc6 = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs, 4)
        )

    def forward(self, x):
        # feature 추출
        features = self.backbone(x)
        
        # 태스크별 분류 
        mise_head = self.backbone.fc1(features)
        pizi_head = self.backbone.fc2(features)
        mosa_head = self.backbone.fc3(features)
        mono_head = self.backbone.fc4(features)
        biddem_head = self.backbone.fc5(features)
        talmo_head = self.backbone.fc6(features)
        
        return mise_head, pizi_head, mosa_head, mono_head, biddem_head, talmo_head
