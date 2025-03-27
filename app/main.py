from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import os

# 🔽 your_model과 inference 불러오기
from app.model.your_model import MultiTaskMobileNetV3  # 모델 아키텍처 정의된 파일에서 불러오기
from app.model.inference import data_transforms, predict_image

# ========================
# FastAPI 앱 초기화
# ========================
app = FastAPI()

# CORS 설정 (필요 시 도메인 제한 가능)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# 모델 로드
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 절대 경로로 모델 경로 설정 (app 폴더가 MTL 폴더 안에 있음)
model_path = os.path.join(os.path.dirname(__file__), "MTL", "app", "model", "MTL_BASIS.pth")

# 모델 로드
model = None
try:
    model = MultiTaskMobileNetV3()  # your_model.py에서 정의된 모델 아키텍처 불러오기
    model.load_state_dict(torch.load(model_path, map_location=device))  # 학습된 가중치 로드
    model.eval()  # 모델을 평가 모드로 전환
except Exception as e:
    print(f"모델 로딩 실패: {e}")
    # 앱이 종료되지 않도록 예외를 던지기
    raise Exception("모델 로딩에 실패했습니다.")

# ========================
# 기본 테스트용 엔드포인트
# ========================
@app.get("/")
def read_root():
    return {"message": "🧠 MTL AI API is running!"}

# ========================
# 예측 엔드포인트
# ========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = data_transforms(image)  # 이미지를 변환

        # 모델을 사용하여 예측
        results = predict_image(model, image, device=device)

        return {"results": results}

    except Exception as e:
        return {"error": f"An error occurred during prediction: {str(e)}"}
