from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "NeuralTool API running on Render 🚀"}
