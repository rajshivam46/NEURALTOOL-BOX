import json
import os
import threading

STATS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'stats.json')
_lock = threading.Lock()

def _load_stats():
    if not os.path.exists(STATS_FILE):
        return {
            "total_datasets": 0,
            "models_trained": 0,
            "total_epochs": 0,
            "sum_accuracy": 0.0,
            "performance_history": []
        }
    with open(STATS_FILE, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {
                "total_datasets": 0,
                "models_trained": 0,
                "total_epochs": 0,
                "sum_accuracy": 0.0,
                "performance_history": []
            }

def _save_stats(stats):
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=4)

def get_dashboard_stats():
    with _lock:
        s = _load_stats()
        avg_acc = (s["sum_accuracy"] / s["models_trained"]) if s["models_trained"] > 0 else 0.0
        return {
            "total_datasets": s["total_datasets"],
            "models_trained": s["models_trained"],
            "total_epochs": s["total_epochs"],
            "avg_accuracy": round(avg_acc * 100, 2), # Return as percentage
            "performance_history": s.get("performance_history", [])
        }

def increment_datasets():
    with _lock:
        s = _load_stats()
        s["total_datasets"] += 1
        _save_stats(s)

def record_model_run(model_name: str, epochs: int, final_accuracy: float, final_loss: float):
    with _lock:
        s = _load_stats()
        s["models_trained"] += 1
        s["total_epochs"] += epochs
        s["sum_accuracy"] += final_accuracy
        
        # Append history
        if "performance_history" not in s:
            s["performance_history"] = []
            
        s["performance_history"].append({
            "name": model_name,
            "val1": int(final_accuracy * 100), # Using val1 for precision/accuracy mapping in charting
            "val2": int(final_loss * 50),      # Scaled for bar chart visibility
            "area": int(final_loss * -100)     # Inverse scaled for area chart visual drop
        })
        
        # Keep only the last 15
        s["performance_history"] = s["performance_history"][-15:]
        
        _save_stats(s)
