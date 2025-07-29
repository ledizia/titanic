import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any
from src.preprocess_data import Preprocessor
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from prometheus_client import Counter, generate_latest, Gauge
from starlette.responses import PlainTextResponse

from pythonjsonlogger import jsonlogger

logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(
    fmt='%(levelname)s %(asctime)s %(name)s %(message)s %(lineno)d %(pathname)s %(funcName)s'
)
logHandler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)
logger.propagate = False 
MODEL_PATH = "src/models/titanic_model.pkl"

model = None


# In a real production environment, this would be a database (e.g., PostgreSQL, SQLite, MongoDB).
prediction_history: List[Dict[str, Any]] = []

# Counter for total prediction requests
PREDICTION_REQUESTS_TOTAL = Counter(
    'titanic_prediction_requests_total',
    'Total prediction requests for the Titanic API'
)

# Counter for prediction errors with labels for categorization
PREDICTION_ERRORS_TOTAL = Counter(
    'titanic_prediction_errors_total',
    'Total errors occurred during predictions in the Titanic API',
    ['error_type', 'status_code']
)

# Gauge for the number of predictions in history (example of state metric)
PREDICTION_HISTORY_COUNT = Gauge(
    'titanic_prediction_history_count',
    'Number of predictions stored in history'
)

class PassengerData(BaseModel):
    Pclass: int = Field(..., description="Passenger class (1, 2, 3)", ge=1, le=3)
    Name: str = Field(..., description="Passenger name")
    Sex: str = Field(..., description="Passenger sex (male, female)")
    Age: float = Field(..., description="Passenger age", ge=0, le=100)
    SibSp: int = Field(..., description="Number of siblings/spouses aboard", ge=0)
    Parch: int = Field(..., description="Number of parents/children aboard", ge=0)
    Ticket: str = Field(..., description="Ticket number")
    Fare: float = Field(..., description="Fare paid", ge=0)
    Cabin: str = Field(..., description="Cabin number (can be empty or 'NaN')", json_schema_extra={"example": "C85"})
    Embarked: str = Field(..., description="Port of embarkation (C, Q, S)")
    
    @validator('Sex')
    def validate_sex(cls, v):
        if v not in ['male', 'female']:
            raise ValueError('Sex must be either "male" or "female"')
        return v
    
    @validator('Embarked')
    def validate_embarked(cls, v):
        if v not in ['C', 'Q', 'S']:
            raise ValueError('Embarked must be one of: C, Q, S')
        return v

class LoadModelRequest(BaseModel):
    model_path: str = Field(..., description="Path to the .pkl file of the new model")
    
    @validator('model_path')
    def validate_model_path(cls, v):
        if not v or not v.strip():
            raise ValueError('model_path cannot be empty')
        return v

def load_artifact(model_path: str):
    """
    Loads the model from the .pkl file.
    """
    global model
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully from: {model_path}")
    except FileNotFoundError as e:
        PREDICTION_ERRORS_TOTAL.labels(error_type="model_not_found", status_code="404").inc()
        logger.error(f"Error loading artifacts: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        PREDICTION_ERRORS_TOTAL.labels(error_type="model_load_error", status_code="500").inc()
        logger.error(f"Error loading artifacts: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading artifacts: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application."""
    # Startup
    logger.info("Starting API... Loading model...")
    load_artifact(MODEL_PATH)
    if model is None:
        logger.critical("Failed to load model during initialization. API may not work correctly.")
    yield
    # Shutdown
    logger.info("Shutting down API...")

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="An API to predict survival on the Titanic, with endpoints for prediction, model loading, and history.",
    version="1.0.0",
    lifespan=lifespan
)

# --- API Endpoints ---

@app.get("/health", summary="Check API status")
async def health_check():
    """
    Endpoint for API status check.
    Returns 200 OK if the API is working.
    """
    logger.info("Request to /health received.")
    return {"status": "API is healthy", "model_loaded": model is not None}

@app.get("/metrics", summary="Expose metrics for Prometheus")
async def metrics():
    """
    Endpoint that exposes monitoring metrics in Prometheus format.
    """
    # Updates the Gauge with the current history size
    PREDICTION_HISTORY_COUNT.set(len(prediction_history))
    logger.info("Request to /metrics received. Serving Prometheus metrics.")
    return PlainTextResponse(generate_latest().decode('utf-8'), media_type="text/plain")

@app.post("/predict", summary="Perform a survival prediction")
async def predict(passenger: PassengerData):
    """
    Receives passenger data in JSON format and returns survival prediction.
    - **passenger**: JSON object with passenger data.
    """
    PREDICTION_REQUESTS_TOTAL.inc() # Increment request counter
    
    try:
        if model is None:
            logger.error("Prediction attempt without loaded model.")
            PREDICTION_ERRORS_TOTAL.labels(error_type="model_not_loaded", status_code="503").inc()
            raise HTTPException(status_code=503, detail="Model not loaded. Please load a model first.")

        logger.info(f"Request to /predict received for passenger: {passenger.Name}")
        
        # Convert input data to a Pandas DataFrame
        preprocessor = Preprocessor()
        input_df = pd.DataFrame([passenger.dict()])

        processed_input = preprocessor.preprocess_data(input_df)
        # Perform prediction
        prediction = model.predict(processed_input)[0]
        prediction_proba_result = model.predict_proba(processed_input)[0]
        
        # Handle both numpy arrays and lists
        if hasattr(prediction_proba_result, 'tolist'):
            prediction_proba = prediction_proba_result.tolist()
        else:
            prediction_proba = list(prediction_proba_result)

        # Map numeric prediction to readable label
        prediction_label = "Survived" if prediction == 1 else "Not Survived"

        # Store prediction in history
        timestamp = datetime.now().isoformat()
        prediction_record = {
            "timestamp": timestamp,
            "input_data": passenger.dict(), 
            "prediction": int(prediction),
            "prediction_label": prediction_label,
            "probabilities": prediction_proba
        }
        prediction_history.append(prediction_record)
        logger.info(f"Prediction made: {prediction_label}")

        return {
            "prediction": int(prediction),
            "prediction_label": prediction_label,
            "probabilities": prediction_proba,
            "message": "Prediction completed successfully."
        }
    except ValueError as e:
        PREDICTION_ERRORS_TOTAL.labels(error_type="validation_error", status_code="400").inc()
        logger.error(f"Error in data validation or processing: {e}")
        raise HTTPException(status_code=400, detail=f"Error in input data: {e}")
    except HTTPException:
        # Re-raise HTTPExceptions without incrementing counters 
        raise
    except Exception as e:
        PREDICTION_ERRORS_TOTAL.labels(error_type="internal_error", status_code="500").inc()
        logger.error(f"Unexpected error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.post("/load", summary="Load a new model")
async def load_new_model(request: LoadModelRequest):
    """
    Allows loading a new model from file paths.
    - **model_path**: Complete path to the .pkl file of the model.
    """
    logger.info(f"Request to /load received. Attempting to load model from: {request.model_path}")
    try:
        load_artifact(request.model_path)
        return {"status": "success", "message": "New model loaded successfully."}
    except HTTPException as e:
        # Track errors based on status code
        if e.status_code >= 400 and e.status_code < 500:
            PREDICTION_ERRORS_TOTAL.labels(error_type="client_error", status_code=str(e.status_code)).inc()
        elif e.status_code >= 500:
            PREDICTION_ERRORS_TOTAL.labels(error_type="server_error", status_code=str(e.status_code)).inc()
        # Re-raise HTTPException so FastAPI can catch it and return the appropriate error
        raise e
    except Exception as e:
        PREDICTION_ERRORS_TOTAL.labels(error_type="load_error", status_code="500").inc()
        logger.error(f"Error loading new model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error loading new model: {e}")


@app.get("/history", summary="Returns the history of prediction calls")
async def get_history():
    """
    Returns the history of all prediction calls made to the API.
    """
    logger.info("Request to /history received.")
    return {"history": prediction_history}
