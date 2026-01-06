from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from . import auth
from . import database
from . import prediction
import os

# Create DB tables
database.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="Skin Disease Prediction API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(prediction.router, tags=["prediction"])

@app.get("/health")
def health_check():
    return {"status": "ok"}

# Mount frontend
# Adjust path to work when running from backend directory
frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="static")
else:
    print(f"Frontend directory not found at {frontend_path}. API only mode.")
