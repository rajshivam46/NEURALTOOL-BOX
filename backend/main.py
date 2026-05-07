from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from backend.api import data_routes, model_routes, predict_routes, cnn_routes
import os

app = FastAPI(title="Neural Network Toolbox API")

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data_routes.router, prefix="/api/data", tags=["Data"])
app.include_router(model_routes.router, prefix="/api/models", tags=["Models"])
app.include_router(predict_routes.router, prefix="/api/predict", tags=["Predict"])
app.include_router(cnn_routes.router, prefix="/api/cnn", tags=["Computer Vision"])

# Ensure frontend directory exists
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dist")
os.makedirs(FRONTEND_DIR, exist_ok=True)

# Mount the static files (js, css, assets)
app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIR, "assets")), name="assets")

# Catch-all route to serve the React app
@app.api_route("/{full_path:path}")
def catch_all(full_path: str):
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Welcome to the Neural Network Toolbox API - Frontend build not found. Please run npm run build in the frontend directory."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
