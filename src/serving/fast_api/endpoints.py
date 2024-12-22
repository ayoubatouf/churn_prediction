import time
from typing import Any, Dict
import psutil
from fastapi import APIRouter, HTTPException, status
from config.config import PROD_LOG_PATH
from src.serving.fast_api.utils import (
    InputData,
    append_to_csv,
    initialize_log_file,
    model,
    process_prediction,
    scaler,
)
import logging

router = APIRouter()


@router.options("/predict/")
async def options_predict() -> Dict[str, str]:
    return {}


@router.post("/predict/", status_code=status.HTTP_200_OK)
async def predict(input_data: InputData) -> Dict[str, Any]:
    initialize_log_file()

    if model is None or scaler is None:
        logging.error("Model or scaler is not loaded.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model or scaler not loaded.",
        )

    start_time = time.time()
    initial_memory = psutil.virtual_memory().percent

    try:
        results, log_entries = await process_prediction(input_data, start_time)

        append_to_csv(log_entries, PROD_LOG_PATH)

        return {"results": results}

    except ValueError as ve:
        logging.error(f"Validation error: {ve}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logging.error(f"Unexpected error during prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred.",
        )
