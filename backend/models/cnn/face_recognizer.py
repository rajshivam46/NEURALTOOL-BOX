import os
import cv2
import numpy as np
import base64
import json
import pandas as pd
from datetime import datetime

# We use the built-in Haar Cascade for 100% native detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Local datastores
EMBEDDINGS_FILE = os.path.join(os.path.dirname(__file__), "registered_faces.json")
MODEL_FILE = os.path.join(os.path.dirname(__file__), "lbph_model.yml")
ATTENDANCE_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "attendance.csv"))

def init_stores():
    if not os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, 'w') as f:
            json.dump({}, f)
    if not os.path.exists(ATTENDANCE_FILE):
        df = pd.DataFrame(columns=["Name", "Timestamp", "Confidence"])
        df.to_csv(ATTENDANCE_FILE, index=False)

init_stores()

def _base64_to_gray_face(b64_str):
    if ',' in b64_str:
        b64_str = b64_str.split(',')[1]
    img_data = base64.b64decode(b64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect the face bounding box (Strict)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    
    # Fallback to a highly lenient detection if the strict one fails (due to backlighting)
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(30, 30))
        
    if len(faces) == 0:
        raise Exception("No human face detected! Please ensure your face is well-lit and clearly visible.")
    
    # Extract the largest face
    (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    return gray[y:y+h, x:x+w]

def register_face(name: str, base64_image: str):
    try:
        face_roi = _base64_to_gray_face(base64_image)
        face_roi = cv2.resize(face_roi, (200, 200)) # Normalize
        
        with open(EMBEDDINGS_FILE, 'r') as f:
            db = json.load(f)
            
        # Give this name a unique integer ID
        if name not in db:
            label_id = len(db) + 1
            db[name] = label_id
        else:
            label_id = db[name]
            
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Load existing trained model if it exists
        if os.path.exists(MODEL_FILE):
            recognizer.read(MODEL_FILE)
            recognizer.update([face_roi], np.array([label_id]))
        else:
            recognizer.train([face_roi], np.array([label_id]))
            
        recognizer.write(MODEL_FILE)
        
        with open(EMBEDDINGS_FILE, 'w') as f:
            json.dump(db, f)
            
        return {"success": True, "message": f"{name} registered successfully!"}
    except Exception as e:
        raise Exception(f"Failed to register face: {str(e)}")

def recognize_and_log(base64_image: str):
    try:
        if not os.path.exists(MODEL_FILE):
             return {"match": False, "message": "No registered users in the database."}
             
        face_roi = _base64_to_gray_face(base64_image)
        face_roi = cv2.resize(face_roi, (200, 200))
        
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(MODEL_FILE)
        
        label_id, distance = recognizer.predict(face_roi)
        
        # LBPH distance: 0 is perfect. Usually < 70 is a strong match.
        if distance > 85:
            return {"match": False, "message": f"Face detected but not recognized tightly. Dist: {distance:.1f}"}
            
        with open(EMBEDDINGS_FILE, 'r') as f:
            db = json.load(f)
            
        # Reverse lookup name by ID
        best_match = next((n for n, i in db.items() if i == label_id), "Unknown")
        
        # Mark Attendance
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Convert distance to a subjective percentage
        conf = max(0, min(100, round(100 - (distance / 1.5), 2)))
        
        row = pd.DataFrame([[best_match, ts, conf]], columns=["Name", "Timestamp", "Confidence"])
        row.to_csv(ATTENDANCE_FILE, mode='a', header=False, index=False)
        
        return {"match": True, "name": best_match, "confidence": conf, "timestamp": ts, 
                "message": f"Welcome, {best_match}!"}
                
    except Exception as e:
         raise Exception(f"Identification failed: {str(e)}")

def get_attendance_logs():
    if not os.path.exists(ATTENDANCE_FILE):
        return []
    df = pd.read_csv(ATTENDANCE_FILE)
    return df.to_dict(orient="records")
