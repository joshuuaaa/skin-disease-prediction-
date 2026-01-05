from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from . import auth, database, prediction
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
# Assuming we run from project root, so frontend is at ./frontend
if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="static")
else:
    print("Frontend directory not found. API only mode.")
