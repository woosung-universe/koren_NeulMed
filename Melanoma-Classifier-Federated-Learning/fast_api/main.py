# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from model_loader import load_model
from utils import preprocess_image

app = FastAPI()
model = load_model("../workspace/clientResults/base_model072.h5")

@app.get("/")
def home():
    return {"message": "Melanoma Prediction API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = preprocess_image(contents)
        pred = model.predict(image)[0][0]
        confidence = round(float(pred) * 100, 2)
        result = "high risk" if confidence > 50 else "low risk"
        return JSONResponse(content={
            "risk": result,
            "confidence": confidence
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)