import torch
from PIL import Image
from torchvision import transforms
import os

# (필요하다면) 같은 폴더에 있는 your_model.py에서 클래스 불러오기
# from .your_model import MultiTaskMobileNetV3

# 전처리 파이프라인(학습 시 사용했던 것과 동일하게)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model(model_path, device='cpu'):
    """
    저장된 .pth 모델 파일을 불러와서 평가 모드로 전환한 뒤 반환
    """
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model

def predict_image(model, image_path, device='cpu'):
    """
    단일 이미지를 입력받아 질환 분류 결과(심각도 포함)를 리턴
    """
    image = Image.open(image_path).convert("RGB")
    input_tensor = data_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        mise_head, pizi_head, mosa_head, mono_head, biddem_head, talmo_head = model(input_tensor)

    # 6개 태스크의 결과를 합침
    outputs = torch.stack([mise_head, pizi_head, mosa_head, mono_head, biddem_head, talmo_head], dim=1)
    probabilities = torch.softmax(outputs, dim=2)
    predictions = torch.argmax(probabilities, dim=2)

    # 질환 이름 리스트 (학습 시 정의했던 순서와 동일해야 함)
    disease_names = ["미세각질", "피지과다", "모낭사이홍반", "모낭농포", "비듬", "탈모"]
    # 심각도 매핑
    disease_severity = {0: "양호", 1: "경증", 2: "중등도", 3: "중증"}

    preds = predictions.cpu().numpy()[0]
    probs = probabilities.cpu().numpy()[0]

    # 최종 결과 구성
    results = []
    for i, (name, pred) in enumerate(zip(disease_names, preds)):
        severity = disease_severity[pred]
        confidence = probs[i, pred] * 100
        results.append({
            "disease": name,
            "severity": severity,
            "confidence": f"{confidence:.2f}%"
        })

    return results

# 실행 예시 (직접 테스트 시)
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = os.path.join(os.path.dirname(__file__), "MTL_BASIS.pth")
    model = load_model(model_path, device=device)
    
    test_image_path = r"테스트_이미지_경로.jpg"
    predictions = predict_image(model, test_image_path, device=device)
    for pred in predictions:
        print(pred)
