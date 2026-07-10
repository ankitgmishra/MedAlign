from fastapi import APIRouter, UploadFile, File
import shutil
import os
from app.utils.response import api_response

router = APIRouter(tags=["upload"])

@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    os.makedirs("datasets", exist_ok=True)
    file_path = f"datasets/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return api_response(
        success=True,
        message=f"Dataset uploaded successfully.",
        data={"file_path": file_path}
    )
