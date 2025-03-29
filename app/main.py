from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import os

# 모델과 데이터셋 불러오기
from app.model.model import MultiTaskMobileNetV3  # 모델 임포트
from app.model.inference import data_transforms, predict_image  # 추론 관련 함수

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 로드
device = torch.device("cpu")  # CPU로 강제 설정

# 모델 경로 설정 (경로가 올바른지 확인)
model_path = os.path.join(os.path.dirname(__file__), "model", "MTL_BASIS.pth")
print(f"모델 경로: {model_path}")  # 디버깅용

model = None
try:
    # 모델 클래스 임포트
    from app.model.model import MultiTaskMobileNetV3  # 클래스 임포트

    # 모델 로드
    model = torch.load(model_path, map_location=device)  # 모델을 CPU로 로드
    model.eval()  # 모델을 평가 모드로 설정
    print(f"모델이 성공적으로 로드되었습니다.")
except FileNotFoundError as e:
    print(f"모델 파일을 찾을 수 없습니다: {e}")
    raise Exception("모델 파일을 찾을 수 없습니다.")
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
