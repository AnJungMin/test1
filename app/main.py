from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch

# 🔽 your_model과 inference 불러오기
from app.model.your_model import MultiTaskMobileNetV3
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
model_path = "app/model/MTL_BASIS.pth"

# safe_globals를 사용하여 모델을 안전하게 로드
try:
    from torch.serialization import safe_globals
    with safe_globals(["MultiTaskMobileNetV3"]):  # 사용자 정의 모델 클래스를 글로벌로 추가
        model = torch.load(model_path, map_location=device)
except Exception as e:
    print(f"모델 로딩 실패: {e}")

model.eval()

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
        image = data_transforms(image)

        results = predict_image(model, image, device=device)
        return {"results": results}

    except Exception as e:
        return {"error": str(e)}
