from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from backend.session_store import get_session

router = APIRouter()

class PredictRequest(BaseModel):
    session_id: str
    feature_values: list[float]

@router.post("/perceptron")
async def predict_perceptron(req: PredictRequest):
    session = get_session(req.session_id)
    
    if 'p_model' not in session:
        raise HTTPException(status_code=404, detail="Perceptron model not trained in this session.")
        
    model = session['p_model']
    targets = session['p_targets']
    
    input_arr = np.array(req.feature_values)
    
    try:
        pred_bin = model.predict(input_arr)[0]
        
        # Determine actual label
        if len(targets) == 2 and not set(targets).issubset({0, 1}):
            final_class = targets[0] if pred_bin == 0 else targets[1]
        else:
            final_class = pred_bin
            
        return {"prediction": float(final_class) if isinstance(final_class, (int, float, np.integer, np.floating)) else final_class}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/backprop")
async def predict_backprop(req: PredictRequest):
    session = get_session(req.session_id)
    
    if 'mlp_model' not in session:
        raise HTTPException(status_code=404, detail="MLP model not trained in this session.")
        
    model = session['mlp_model']
    targets = session['mlp_targets']
    x_mean = np.array(session['mlp_mean'])
    x_std = np.array(session['mlp_std'])
    
    try:
        input_arr = np.array(req.feature_values).reshape(1, -1)
        scaled_input = (input_arr - x_mean) / x_std
        
        pred_prob = model.forward(scaled_input)[0][0]
        pred_bin = 1 if pred_prob >= 0.5 else 0
        
        if len(targets) == 2 and not set(targets).issubset({0, 1}):
            final_class = targets[0] if pred_bin == 0 else targets[1]
        else:
            final_class = pred_bin
            
        return {
            "prediction": float(final_class) if isinstance(final_class, (int, float, np.integer, np.floating)) else final_class,
            "confidence": float(pred_prob)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/rnn")
async def predict_rnn(req: PredictRequest):
    session = get_session(req.session_id)
    
    if 'rnn_model' not in session:
        raise HTTPException(status_code=404, detail="RNN model not trained in this session.")
        
    model = session['rnn_model']
    s_m = session['rnn_mean']
    s_s = session['rnn_std']
    
    try:
        seq_input = np.array(req.feature_values).reshape(-1, 1)
        scaled_input = (seq_input - s_m) / s_s
        
        pred_scaled = model.predict(scaled_input)[0][0]
        pred_actual = (pred_scaled * s_s) + s_m
        
        return {"prediction": float(pred_actual)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
