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
    """
    Passenger data model for Titanic survival prediction.
    
    This model defines the structure and validation rules for passenger data
    used in survival predictions. All fields are required and have specific
    validation constraints.
    
    **Field Descriptions:**
    - `Pclass`: Passenger class (1=First, 2=Second, 3=Third)
    - `Name`: Passenger's full name (used for title extraction)
    - `Sex`: Gender (male/female)
    - `Age`: Age in years (0-100)
    - `SibSp`: Number of siblings/spouses aboard (0+)
    - `Parch`: Number of parents/children aboard (0+)
    - `Ticket`: Ticket number (string)
    - `Fare`: Fare paid (0+)
    - `Cabin`: Cabin number (optional, can be empty)
    - `Embarked`: Port of embarkation (C=Cherbourg, Q=Queenstown, S=Southampton)
    
    **Validation Rules:**
    - Pclass must be 1, 2, or 3
    - Age must be between 0 and 100
    - SibSp and Parch must be non-negative
    - Fare must be non-negative
    - Sex must be 'male' or 'female'
    - Embarked must be 'C', 'Q', or 'S'
    
    **Example:**
    ```json
    {
        "Pclass": 3,
        "Name": "Braund, Mr. Owen Harris",
        "Sex": "male",
        "Age": 22.0,
        "SibSp": 1,
        "Parch": 0,
        "Ticket": "A/5 21171",
        "Fare": 7.25,
        "Cabin": "",
        "Embarked": "S"
    }
    ```
    """
    Pclass: int = Field(
        ..., 
        description="Passenger class (1=First, 2=Second, 3=Third)", 
        ge=1, 
        le=3,
        example=3
    )
    Name: str = Field(
        ..., 
        description="Passenger's full name (used for title extraction)",
        example="Braund, Mr. Owen Harris"
    )
    Sex: str = Field(
        ..., 
        description="Passenger gender (male/female)",
        example="male"
    )
    Age: float = Field(
        ..., 
        description="Passenger age in years", 
        ge=0, 
        le=100,
        example=22.0
    )
    SibSp: int = Field(
        ..., 
        description="Number of siblings/spouses aboard", 
        ge=0,
        example=1
    )
    Parch: int = Field(
        ..., 
        description="Number of parents/children aboard", 
        ge=0,
        example=0
    )
    Ticket: str = Field(
        ..., 
        description="Ticket number",
        example="A/5 21171"
    )
    Fare: float = Field(
        ..., 
        description="Fare paid", 
        ge=0,
        example=7.25
    )
    Cabin: str = Field(
        ..., 
        description="Cabin number (can be empty or 'NaN')", 
        json_schema_extra={"example": "C85"},
        example=""
    )
    Embarked: str = Field(
        ..., 
        description="Port of embarkation (C=Cherbourg, Q=Queenstown, S=Southampton)",
        example="S"
    )
    
    @validator('Sex')
    def validate_sex(cls, v):
        """Validate that sex is either 'male' or 'female'."""
        if v not in ['male', 'female']:
            raise ValueError('Sex must be either "male" or "female"')
        return v
    
    @validator('Embarked')
    def validate_embarked(cls, v):
        """Validate that embarked port is valid."""
        if v not in ['C', 'Q', 'S']:
            raise ValueError('Embarked must be one of: C, Q, S')
        return v

class LoadModelRequest(BaseModel):
    """
    Model loading request for dynamic model updates.
    
    This model defines the structure for loading new machine learning models
    without restarting the API. The model path must point to a valid .pkl file.
    
    **Field Descriptions:**
    - `model_path`: Complete file path to the model file (.pkl format)
    
    **Validation Rules:**
    - model_path cannot be empty or whitespace-only
    - model_path should point to a valid .pkl file
    - File must exist and be readable
    
    **Example:**
    ```json
    {
        "model_path": "/path/to/new_titanic_model.pkl"
    }
    ```
    
    **Supported Formats:**
    - Pickle (.pkl) files containing scikit-learn models
    - Models must have `predict()` and `predict_proba()` methods
    """
    model_path: str = Field(
        ..., 
        description="Complete file path to the model file (.pkl format)",
        example="/path/to/new_titanic_model.pkl"
    )
    
    @validator('model_path')
    def validate_model_path(cls, v):
        """Validate that model path is not empty."""
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

@app.get(
    "/health", 
    summary="Health Check",
    description="Check the health status of the Titanic Survival Prediction API",
    response_description="API health status and model loading state",
    tags=["Monitoring"],
    responses={
        200: {
            "description": "API is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "API is healthy",
                        "model_loaded": True
                    }
                }
            }
        }
    }
)
async def health_check():
    """
    Check the health status of the Titanic Survival Prediction API.
    
    This endpoint provides information about:
    - API operational status
    - Model loading state
    
    **Returns:**
    - `status`: Current API status message
    - `model_loaded`: Boolean indicating if the ML model is loaded and ready
    
    **Use Cases:**
    - Health monitoring and alerting
    - Load balancer health checks
    - Service discovery
    
    **Example Response:**
    ```json
    {
        "status": "API is healthy",
        "model_loaded": true
    }
    ```
    """
    logger.info("Request to /health received.")
    return {"status": "API is healthy", "model_loaded": model is not None}

@app.get(
    "/metrics", 
    summary="Prometheus Metrics",
    description="Expose monitoring metrics in Prometheus format for observability",
    response_description="Prometheus-formatted metrics",
    tags=["Monitoring"],
    responses={
        200: {
            "description": "Prometheus metrics",
        }
    }
)
async def metrics():
    """
    Expose monitoring metrics in Prometheus format for observability.
    
    This endpoint provides comprehensive metrics for monitoring the API:
    - **Request Metrics**: Total number of prediction requests
    - **Error Metrics**: Categorized errors with labels (error_type, status_code)
    - **State Metrics**: Current prediction history count
    
    **Available Metrics:**
    - `titanic_prediction_requests_total`: Counter for total requests
    - `titanic_prediction_errors_total`: Counter for errors with labels
    - `titanic_prediction_history_count`: Gauge for history size
    
    **Error Categories:**
    - `validation_error`: Data validation errors (400)
    - `internal_error`: Unexpected server errors (500)
    - `model_not_loaded`: Model not available (503)
    - `model_not_found`: Model file not found (404)
    - `model_load_error`: Model loading failures (500)
    
    **Use Cases:**
    - Prometheus monitoring integration
    - Grafana dashboard data source
    - Alerting and alerting rules
    - Performance monitoring
    
    **Example Usage:**
    ```bash
    curl http://localhost:8000/metrics
    ```
    """
    # Updates the Gauge with the current history size
    PREDICTION_HISTORY_COUNT.set(len(prediction_history))
    logger.info("Request to /metrics received. Serving Prometheus metrics.")
    return PlainTextResponse(generate_latest().decode('utf-8'), media_type="text/plain")

@app.post(
    "/predict", 
    summary="Predict Survival",
    description="Predict survival probability for a Titanic passenger based on their characteristics",
    response_description="Prediction result with survival probability and confidence scores",
    tags=["Predictions"],
    responses={
        200: {
            "description": "Successful prediction",
            "content": {
                "application/json": {
                    "example": {
                        "prediction": 0,
                        "prediction_label": "Not Survived",
                        "probabilities": [0.7, 0.3],
                        "message": "Prediction completed successfully."
                    }
                }
            }
        },
        400: {
            "description": "Bad request - validation error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Error in input data: Invalid passenger data"
                    }
                }
            }
        },
        422: {
            "description": "Validation error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "loc": ["body", "Sex"],
                                "msg": "value is not a valid enumeration member; permitted: 'male', 'female'",
                                "type": "type_error.enum; enum_values allowed_values"
                            }
                        ]
                    }
                }
            }
        },
        503: {
            "description": "Service unavailable - model not loaded",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Model not loaded. Please load a model first."
                    }
                }
            }
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Internal server error: Unexpected error during prediction"
                    }
                }
            }
        }
    }
)
async def predict(passenger: PassengerData):
    """
    Predict survival probability for a Titanic passenger.
    
    This endpoint uses a trained machine learning model to predict whether a passenger
    would have survived the Titanic disaster based on their characteristics.
    
    **Features:**
    - Real-time prediction using trained ML model
    - Probability scores for both survival and non-survival
    - Automatic feature engineering and preprocessing
    - Prediction history tracking
    - Comprehensive error handling
    
    **Input Parameters:**
    - `Pclass`: Passenger class (1=First, 2=Second, 3=Third)
    - `Name`: Passenger's full name
    - `Sex`: Gender (male/female)
    - `Age`: Age in years (0-100)
    - `SibSp`: Number of siblings/spouses aboard
    - `Parch`: Number of parents/children aboard
    - `Ticket`: Ticket number
    - `Fare`: Fare paid
    - `Cabin`: Cabin number (optional)
    - `Embarked`: Port of embarkation (C=Cherbourg, Q=Queenstown, S=Southampton)
    
    **Returns:**
    - `prediction`: Binary prediction (0=Not Survived, 1=Survived)
    - `prediction_label`: Human-readable prediction label
    - `probabilities`: Probability scores [not_survived, survived]
    - `message`: Success message
    
    **Error Handling:**
    - 400: Data validation or processing errors
    - 422: Input validation errors (invalid field values)
    - 503: Model not loaded
    - 500: Internal server errors
    
    **Example Request:**
    ```json
    {
        "Pclass": 3,
        "Name": "Braund, Mr. Owen Harris",
        "Sex": "male",
        "Age": 22.0,
        "SibSp": 1,
        "Parch": 0,
        "Ticket": "A/5 21171",
        "Fare": 7.25,
        "Cabin": "",
        "Embarked": "S"
    }
    ```
    
    **Example Response:**
    ```json
    {
        "prediction": 0,
        "prediction_label": "Not Survived",
        "probabilities": [0.7, 0.3],
        "message": "Prediction completed successfully."
    }
    ```
    
    **Model Information:**
    - Uses Random Forest classifier
    - Features include: passenger class, age, family size, title extraction
    - Trained on historical Titanic passenger data
    - Provides both prediction and probability scores
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

@app.post(
    "/load", 
    summary="Load Model",
    description="Load a new machine learning model from a file path",
    response_description="Model loading status and result message",
    tags=["Model Management"],
    responses={
        200: {
            "description": "Model loaded successfully",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "message": "New model loaded successfully."
                    }
                }
            }
        },
        400: {
            "description": "Bad request - invalid model path",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "model_path cannot be empty"
                    }
                }
            }
        },
        404: {
            "description": "Model file not found",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Model not found at: /path/to/model.pkl"
                    }
                }
            }
        },
        422: {
            "description": "Validation error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "loc": ["body", "model_path"],
                                "msg": "field required",
                                "type": "value_error.missing"
                            }
                        ]
                    }
                }
            }
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Error loading new model: Invalid model format"
                    }
                }
            }
        }
    }
)
async def load_new_model(request: LoadModelRequest):
    """
    Load a new machine learning model from a file path.
    
    This endpoint allows dynamic model loading without restarting the API.
    The new model will be used for all subsequent predictions.
    
    **Features:**
    - Dynamic model loading without API restart
    - Support for different model formats (.pkl files)
    - Validation of model file existence
    - Error handling for invalid models
    - Automatic model replacement
    
    **Input Parameters:**
    - `model_path`: Complete file path to the model file (.pkl format)
    
    **Returns:**
    - `status`: Loading status ("success" or "error")
    - `message`: Descriptive message about the loading result
    
    **Error Handling:**
    - 400: Invalid model path (empty or whitespace)
    - 404: Model file not found at specified path
    - 422: Validation errors in request body
    - 500: Model loading errors (corrupted file, invalid format)
    
    **Supported Model Formats:**
    - Pickle (.pkl) files containing scikit-learn models
    - Models must have `predict()` and `predict_proba()` methods
    
    **Example Request:**
    ```json
    {
        "model_path": "/path/to/new_titanic_model.pkl"
    }
    ```
    
    **Example Response:**
    ```json
    {
        "status": "success",
        "message": "New model loaded successfully."
    }
    ```
    
    **Use Cases:**
    - Model updates without downtime
    - A/B testing different models
    - Model version management
    - Hot-swapping improved models
    
    **Security Considerations:**
    - Validate model file integrity
    - Ensure model compatibility
    - Monitor model performance after loading
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


@app.get(
    "/history", 
    summary="Prediction History",
    description="Retrieve the history of all prediction calls made to the API",
    response_description="List of prediction records with timestamps and results",
    tags=["Monitoring"],
    responses={
        200: {
            "description": "Prediction history",
            "content": {
                "application/json": {
                    "example": {
                        "history": [
                            {
                                "timestamp": "2024-01-15T10:30:45.123456",
                                "input_data": {
                                    "Pclass": 3,
                                    "Name": "Braund, Mr. Owen Harris",
                                    "Sex": "male",
                                    "Age": 22.0,
                                    "SibSp": 1,
                                    "Parch": 0,
                                    "Ticket": "A/5 21171",
                                    "Fare": 7.25,
                                    "Cabin": "",
                                    "Embarked": "S"
                                },
                                "prediction": 0,
                                "prediction_label": "Not Survived",
                                "probabilities": [0.7, 0.3]
                            }
                        ]
                    }
                }
            }
        }
    }
)
async def get_history():
    """
    Retrieve the history of all prediction calls made to the API.
    
    This endpoint provides access to the complete prediction history, including
    input data, predictions, and metadata for each request.
    
    **Features:**
    - Complete prediction history with timestamps
    - Original input data for each prediction
    - Prediction results and probability scores
    - Chronological ordering of predictions
    - In-memory storage (resets on API restart)
    
    **Returns:**
    - `history`: Array of prediction records, each containing:
        - `timestamp`: ISO format timestamp of the prediction
        - `input_data`: Original passenger data used for prediction
        - `prediction`: Binary prediction result (0/1)
        - `prediction_label`: Human-readable prediction ("Survived"/"Not Survived")
        - `probabilities`: Probability scores [not_survived, survived]
    
    **History Record Structure:**
    ```json
    {
        "timestamp": "2024-01-15T10:30:45.123456",
        "input_data": {
            "Pclass": 3,
            "Name": "Braund, Mr. Owen Harris",
            "Sex": "male",
            "Age": 22.0,
            "SibSp": 1,
            "Parch": 0,
            "Ticket": "A/5 21171",
            "Fare": 7.25,
            "Cabin": "",
            "Embarked": "S"
        },
        "prediction": 0,
        "prediction_label": "Not Survived",
        "probabilities": [0.7, 0.3]
    }
    ```
    
    **Use Cases:**
    - Audit trail for predictions
    - Model performance analysis
    - Debugging prediction issues
    - Data analysis and reporting
    - Compliance and governance
    
    **Limitations:**
    - History is stored in memory (not persistent)
    - History is cleared on API restart
    - No pagination (returns all history)
    - No filtering or search capabilities
    
    **Example Response:**
    ```json
    {
        "history": [
            {
                "timestamp": "2024-01-15T10:30:45.123456",
                "input_data": {...},
                "prediction": 0,
                "prediction_label": "Not Survived",
                "probabilities": [0.7, 0.3]
            },
            {
                "timestamp": "2024-01-15T10:31:12.456789",
                "input_data": {...},
                "prediction": 1,
                "prediction_label": "Survived",
                "probabilities": [0.2, 0.8]
            }
        ]
    }
    ```
    
    **Note:** For production use, consider implementing persistent storage
    (database) and pagination for large history datasets.
    """
    logger.info("Request to /history received.")
    return {"history": prediction_history}
