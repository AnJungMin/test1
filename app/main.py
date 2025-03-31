import os
import sys
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.model.model import MultiTaskMobileNetV3
from app.model.inference import predict_image  # 🔥 data_transforms는 필요 없음
import io

# 🔐 클래스 등록 (렌더 호환)
sys.modules["__main__"].MultiTaskMobileNetV3 = MultiTaskMobileNetV3

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 디바이스 설정
device = torch.device("cpu")
model = None
model_path = os.path.join(os.path.dirname(__file__), "model", "MTL_BASIS.pth")
print(f"모델 경로: {model_path}")

# 모델 로딩
try:
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    print("✅ 모델 로딩 완료")
except Exception as e:
    model = None
    print(f"❌ 모델 로딩 실패: {e}")

@app.get("/")
def read_root():
    if model is None:
        return {"status": "❌ 모델 미로딩"}
    return {"message": "🧠 MTL AI API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "❌ 모델이 로딩되지 않았습니다."}
    try:
        contents = await file.read()

        # ✅ 수정된 부분: 이미지 내용을 그대로 전달
        results = predict_image(model, contents, device=device)

        return {"results": results}
    except Exception as e:
        return {"error": f"🔥 예측 중 오류 발생: {str(e)}"}
