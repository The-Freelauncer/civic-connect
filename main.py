from fastapi import FastAPI, File, UploadFile, Request 
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import numpy as np
from PIL import Image
import tensorflow as tf
from io import BytesIO
import os





# -------------------------
# Create FastAPI app
# -------------------------
app = FastAPI()

# -------------------------
# CORS middleware
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Setup static and templates
# -------------------------

from pathlib import Path

# -------------------------
# Setup static and templates
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# -------------------------
# Routes to serve HTML pages
# -------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        return HTMLResponse(f"<h1>Error loading template:</h1><pre>{e}</pre>")


@app.get("/login", response_class=HTMLResponse)
async def login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/feed", response_class=HTMLResponse)
async def feed(request: Request):
    return templates.TemplateResponse("feed.html", {"request": request})

@app.get("/UserDashboard", response_class=HTMLResponse)
async def UserDashboard(request: Request):
    return templates.TemplateResponse("UserDashboard.html", {"request": request})

@app.get("/tracker", response_class=HTMLResponse)
async def tracker(request: Request):
    return templates.TemplateResponse("tracker.html", {"request": request})

@app.get("/realtime", response_class=HTMLResponse)
async def realtime(request: Request):
    return templates.TemplateResponse("realtime.html", {"request": request})


@app.get("/officerpage", response_class=HTMLResponse)
async def officerpage(request: Request):
    return templates.TemplateResponse("officerpage.html", {"request": request})

@app.get("/Reporter", response_class=HTMLResponse)
async def Reporter(request: Request):
    return templates.TemplateResponse("Reporter.html", {"request": request})

@app.get("/admin", response_class=HTMLResponse)
async def admin(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})

@app.get("/Acknowledgement", response_class=HTMLResponse)
async def Acknowledgement(request: Request):
    return templates.TemplateResponse("Acknowledgement.html", {"request": request})





# -------------------------
# Health check endpoint
# -------------------------
@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

# -------------------------
# Load trained model
# -------------------------
MODEL_PATH = os.path.join("models", "hackathon_ai_model_version8.keras")
try:
    MODEL = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)
    MODEL = None

CLASS_NAMES = ['clean', 'dirty', 'potholes']

# -------------------------
# Helper function to process image
# -------------------------
def read_file_as_image(data: bytes):
    """Convert uploaded file into a processed image array and predict."""
    try:
        image = Image.open(BytesIO(data)).convert("RGB")
        image = image.resize((256, 256))

        img_array = np.array(image)
        img_batch = np.expand_dims(img_array, axis=0)

        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        return {"class": predicted_class, "confidence": confidence}

    except Exception as e:
        return {"error": str(e)}

# -------------------------
# Prediction endpoint
# -------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        print("📸 Received file:", file.filename, "Size:", len(contents))
        if MODEL is None:
            return {"error": "Model not loaded"}
        return read_file_as_image(contents)
    except Exception as e:
        print("❌ Prediction error:", e)
        return {"error": str(e)}


# -------------------------
# Run server
# -------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
