from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import pandas as pd
from io import StringIO
from backend.utils.data_utils import get_numeric_columns
from backend.session_store import get_session
from backend.utils.stats_store import increment_datasets, get_dashboard_stats
import uuid

router = APIRouter()

@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    session_id: str = Form(None)
):
    if not session_id:
        session_id = str(uuid.uuid4())
        
    try:
        content = await file.read()
        df = pd.read_csv(StringIO(content.decode('utf-8')))
        
        # Save dataframe to memory state temporarily if needed
        # but storing the whole DF in memory is fine for a toolbox app.
        session = get_session(session_id)
        session['df'] = df
        
        increment_datasets()
        
        cols = get_numeric_columns(df)
        
        return {
            "session_id": session_id,
            "columns": cols,
            "preview": df.head().to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing dataframe: {str(e)}")

@router.get("/dashboard-stats")
async def get_dashboard():
    return get_dashboard_stats()
