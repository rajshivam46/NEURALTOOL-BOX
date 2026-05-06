from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from backend.session_store import get_session
from backend.models.perceptron.model import SimplePerceptron
from backend.models.backprop.model import MLP
from backend.models.rnn.model import SimpleRNN
from backend.utils.data_utils import validate_binary_target, standardize_data
from backend.utils.stats_store import record_model_run

router = APIRouter()

class PerceptronTrainRequest(BaseModel):
    session_id: str
    target_col: str
    feature_cols: list[str]
    lr: float = 0.1
    epochs: int = 50

@router.post("/train/perceptron")
async def train_perceptron(req: PerceptronTrainRequest):
    session = get_session(req.session_id)
    if 'df' not in session:
        raise HTTPException(status_code=400, detail="Dataframe not uploaded.")
        
    df = session['df']
    
    try:
        X = df[req.feature_cols].values
        y_raw = df[req.target_col].values
        
        is_binary, unique_targets, y = validate_binary_target(y_raw)
        if not is_binary:
            raise HTTPException(status_code=400, detail="Perceptron only supports Binary targets (2 classes).")
            
        model = SimplePerceptron(input_size=X.shape[1], lr=req.lr, epochs=req.epochs)
        errors, accuracies = model.fit(X, y)
        
        # Save model to session
        session['p_model'] = model
        session['p_features'] = req.feature_cols
        session['p_targets'] = unique_targets.tolist()
        
        # Record stats
        final_acc = float(accuracies[-1]) if accuracies else 0.0
        final_loss = float(errors[-1]) if errors else 0.0
        record_model_run("Perceptron", req.epochs, final_acc, final_loss)
        
        return {
            "message": "Training Complete",
            "errors": errors,
            "accuracies": accuracies
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

class BackpropTrainRequest(BaseModel):
    session_id: str
    target_col: str
    feature_cols: list[str]
    hidden_nodes: int = 8
    lr: float = 0.1
    epochs: int = 1000

@router.post("/train/backprop")
async def train_backprop(req: BackpropTrainRequest):
    session = get_session(req.session_id)
    if 'df' not in session:
        raise HTTPException(status_code=400, detail="Dataframe not uploaded.")
        
    df = session['df']
    
    try:
        X_raw = df[req.feature_cols].values
        y_raw = df[req.target_col].values
        
        is_binary, unique_targets, y = validate_binary_target(y_raw)
        y = y.reshape(-1, 1)
        
        X, X_mean, X_std = standardize_data(X_raw)
        
        model = MLP(input_dim=X.shape[1], hidden_dim=req.hidden_nodes, output_dim=1, lr=req.lr)
        losses = []
        accs = []
        
        for i in range(req.epochs):
            model.forward(X)
            loss = model.backward(X, y)
            if i % max(1, req.epochs // 100) == 0 or i == req.epochs - 1:
                preds = model.predict(X)
                acc = np.mean(preds == y)
                losses.append(loss)
                accs.append(acc)
                
        session['mlp_model'] = model
        session['mlp_features'] = req.feature_cols
        session['mlp_targets'] = unique_targets.tolist()
        session['mlp_mean'] = X_mean.tolist()
        session['mlp_std'] = X_std.tolist()
        
        final_acc = float(accs[-1]) if accs else 0.0
        final_loss = float(losses[-1]) if losses else 0.0
        record_model_run("Backprop MLP", req.epochs, final_acc, final_loss)
        
        return {
            "message": "Training Complete",
            "errors": losses,
            "accuracies": accs
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

class RNNTrainRequest(BaseModel):
    session_id: str
    target_col: str
    seq_length: int = 5
    hidden_dim: int = 8
    epochs: int = 100

@router.post("/train/rnn")
async def train_rnn(req: RNNTrainRequest):
    session = get_session(req.session_id)
    if 'df' not in session:
        raise HTTPException(status_code=400, detail="Dataframe not uploaded.")
        
    df = session['df']
    
    try:
        series = df[req.target_col].values
        series_mean = np.mean(series)
        series_std = np.std(series) + 1e-8
        series_norm = (series - series_mean) / series_std
        
        X = []
        Y = []
        for i in range(len(series_norm) - req.seq_length):
            X.append(series_norm[i:i+req.seq_length])
            Y.append(series_norm[i+req.seq_length])
            
        X_train = np.array(X)[..., np.newaxis]
        y_train = np.array(Y)
        
        model = SimpleRNN(input_dim=1, hidden_dim=req.hidden_dim, output_dim=1)
        losses = []
        
        for epoch in range(req.epochs):
            epoch_loss = 0
            idx = np.random.choice(len(X_train), size=min(10, len(X_train)), replace=False)
            for i in idx:
                loss = model.train_step(X_train[i], y_train[i])
                epoch_loss += loss
            losses.append(epoch_loss / len(idx))
            
        session['rnn_model'] = model
        session['rnn_seq_len'] = req.seq_length
        session['rnn_mean'] = float(series_mean)
        session['rnn_std'] = float(series_std)
        
        final_loss = float(losses[-1]) if losses else 0.0
        record_model_run("RNN", req.epochs, 0.0, final_loss) # RNN doesn't track accuracy natively in this implementation
        
        return {
            "message": "Training Complete",
            "errors": losses,  # Reusing errors chart for RNN
            "accuracies": []
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
