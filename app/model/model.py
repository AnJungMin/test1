import torch
import torch.nn as nn
from torchvision import models

class MultiTaskMobileNetV3(nn.Module):
    def __init__(self, pretrained=True):
        super(MultiTaskMobileNetV3, self).__init__()
        # 사전학습된 MobileNetV3-Large 백본 불러오기
        self.backbone = models.mobilenet_v3_large(pretrained=pretrained)
        num_ftrs = self.backbone.classifier[0].in_features
        
        # 마지막 classifier 제거 -> feature vector 추출
        self.backbone.classifier = nn.Identity()
        
        # 각 태스크별 classifier 정의 (6개의 질환 분류)
        self.fc_layers = nn.ModuleList([
            self._create_fc_layer(num_ftrs) for _ in range(6)
        ])

    def _create_fc_layer(self, num_ftrs):
        return nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs),
            nn.Hardswish(),  # 혹은 ReLU 등
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs, 4)  # 클래스 수: 0~3
        )

    def forward(self, x):
        # feature 추출
        features = self.backbone(x)
        
        # 각 태스크별 결과
        task_outputs = [fc_layer(features) for fc_layer in self.fc_layers]
        
        return tuple(task_outputs)
