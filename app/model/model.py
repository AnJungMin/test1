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
        
        # 각 태스크별 classifier 정의
        self.fc_layers = nn.ModuleList([
            self._create_fc_layer(num_ftrs) for _ in range(6)
        ])

    def _create_fc_layer(self, num_ftrs):
        return nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs),
            nn.Hardswish(),  # 활성화 함수 선택 (ReLU로 변경 가능)
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs, 4)  # 4개의 클래스 (0~3)
        )

    def forward(self, x):
        # feature 추출
        features = self.backbone(x)
        
        # 각 태스크별 분류 결과
        task_outputs = [fc_layer(features) for fc_layer in self.fc_layers]
        
        return tuple(task_outputs)

# 모델 생성 예시
# 모델을 사용할 때, pretrained=False로 사전 학습된 가중치를 사용하지 않게 설정할 수도 있습니다.
# model = MultiTaskMobileNetV3(pretrained=True)
