import os
import sys
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

# 모델과 추론 관련 함수 임포트
from app.model.model import MultiTaskMobileNetV3  # 모델 정의
from app.model.inference import data_transforms, predict_image  # 전처리 및 예측 함수

# ⚠️ torch.load 오류 방지: 클래스 경로 등록
sys.modules["__main__"].MultiTaskMobileNetV3 = MultiTaskMobileNetV3

app = FastAPI()

# CORS 설정 (프론트에서 접근 가능하도록)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 디바이스 설정 (렌더는 GPU 미지원 → CPU)
device = torch.device("cpu")

# 모델 경로
model_path = os.path.join(os.path.dirname(__file__), "model", "MTL_BASIS.pth")
print(f"모델 경로: {model_path}")

# 모델 로드
try:
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    print("✅ 모델 로딩 완료")
except Exception as e:
    print(f"❌ 모델 로딩 실패: {e}")
    raise Exception("모델 로딩 중 오류 발생")

# 기본 테스트용 라우트
@app.get("/")
def read_root():
    return {"message": "🧠 MTL AI API is running!"}

# 예측 라우트
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        results = predict_image(model, contents, device=device)
        return {"results": results}
    except Exception as e:
        return {"error": f"An error occurred during prediction: {str(e)}"}
