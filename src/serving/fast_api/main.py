from typing import Dict
from fastapi import FastAPI
from src.serving.fast_api.endpoints import router as predict_router

app = FastAPI()

app.include_router(predict_router, prefix="/api")


@app.get("/")
async def root() -> Dict[str, str]:

    return {"message": "Churn prediction API is running."}
