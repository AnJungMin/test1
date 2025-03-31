import os
import sys
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from app.model.model import MultiTaskMobileNetV3
from app.model.inference import data_transforms, predict_image

# 🔐 모델 클래스를 __main__에 등록 (렌더 호환성)
sys.modules["__main__"].MultiTaskMobileNetV3 = MultiTaskMobileNetV3

app = FastAPI()

# ✅ CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 장치 설정
device = torch.device("cpu")
model = None
model_path = os.path.join(os.path.dirname(__file__), "model", "MTL_BASIS.pth")
print(f"모델 경로: {model_path}")

# ✅ 모델 로딩
try:
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    print("✅ 모델 로딩 완료")
except Exception as e:
    model = None
    print(f"❌ 모델 로딩 실패: {e}")

# ✅ 기본 라우트
@app.get("/")
def read_root():
    if model is None:
        return {"status": "❌ 모델 미로딩"}
    return {"message": "🧠 MTL AI API is running!"}

# ✅ 예측 라우트
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "❌ 모델이 로딩되지 않았습니다."}
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor = data_transforms(image).unsqueeze(0).to(device)

        # 🔥 추론
        results = predict_image(model, tensor, device=device)
        return {"results": results}
    except Exception as e:
        return {"error": f"🔥 예측 중 오류 발생: {str(e)}"}
