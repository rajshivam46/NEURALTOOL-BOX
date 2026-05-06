from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.models.cnn.face_recognizer import register_face, recognize_and_log, get_attendance_logs

router = APIRouter()

class RegisterPayload(BaseModel):
    name: str
    image: str

class RecognizePayload(BaseModel):
    image: str

@router.post("/register")
async def register(req: RegisterPayload):
    try:
        res = register_face(req.name, req.image)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/attendance")
async def mark_attendance(req: RecognizePayload):
    try:
        res = recognize_and_log(req.image)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/logs")
async def fetch_logs():
    try:
        return get_attendance_logs()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
