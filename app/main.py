import os
import sys
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

# 모델과 추론 관련 함수 임포트
from app.model.model import MultiTaskMobileNetV3  # 모델 정의 파일
from app.model.inference import data_transforms, predict_image  # 추론 함수

# __main__ 모듈에 모델 클래스를 등록 (모델 저장 시 __main__에 저장된 경우 대비)
sys.modules["__main__"].MultiTaskMobileNetV3 = MultiTaskMobileNetV3

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 로드 (GPU 대신 CPU 사용)
device = torch.device("cpu")

# 모델 경로 설정 (현재 파일 기준 상대경로)
model_path = os.path.join(os.path.dirname(__file__), "model", "MTL_BASIS.pth")
print(f"모델 경로: {model_path}")

model = None
try:
    from torch.serialization import safe_globals
    # 실제 클래스 객체를 전달하여 안전하게 로드하고, weights_only 옵션을 False로 설정
    with safe_globals([MultiTaskMobileNetV3]):
        model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()  # 모델을 평가 모드로 설정
    print("모델이 성공적으로 로드되었습니다.")
except Exception as e:
    print(f"모델 로딩 실패: {e}")
    raise Exception("모델 로딩에 실패했습니다.")

@app.get("/")
def read_root():
    return {"message": "🧠 MTL AI API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = data_transforms(image)  # 이미지 전처리

        # 모델을 사용하여 예측
        results = predict_image(model, image, device=device)

        return {"results": results}

    except Exception as e:
        return {"error": f"An error occurred during prediction: {str(e)}"}
