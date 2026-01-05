from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import os
import uuid
from models import model_utils

router = APIRouter()

UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save file
    file_extension = file.filename.split('.')[-1]
    filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        label, confidence = model_utils.predict_single_image(file_path)
        
        # Cleanup? Maybe keep for history?
        # os.remove(file_path) 
        
        return {
            "filename": file.filename,
            "prediction": label,
            "confidence": confidence,
            "medical_disclaimer": "This is an AI prediction and NOT a medical diagnosis. Consult a dermatologist."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
