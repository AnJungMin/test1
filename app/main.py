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

# ✅ __main__ 모듈에 모델 클래스 등록 (렌더에서 torch.load 문제 방지용)
sys.modules["__main__"].MultiTaskMobileNetV3 = MultiTaskMobileNetV3

app = FastAPI()

# ✅ CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 모델 로드 (CPU 강제 지정)
device = torch.device("cpu")

# ✅ 모델 경로 설정
model_path = os.path.join(os.path.dirname(__file__), "model", "MTL_BASIS.pth")
print(f"모델 경로: {model_path}")

model = None
try:
    from torch.serialization import safe_globals
    with safe_globals([MultiTaskMobileNetV3]):
        model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    print("모델이 성공적으로 로드되었습니다.")
except Exception as e:
    print(f"모델 로딩 실패: {e}")
    raise Exception("모델 로딩에 실패했습니다.")

# ✅ 기본 라우트
@app.get("/")
def read_root():
    return {"message": "🧠 MTL AI API is running!"}

# ✅ 예측 라우트
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # 🔥 전처리 없이 raw bytes 전달
        results = predict_image(model, contents, device=device)

        return {"results": results}

    except Exception as e:
        return {"error": f"An error occurred during prediction: {str(e)}"}
