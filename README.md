# Titanic Survival Prediction API 🚢

A FastAPI-based machine learning API for predicting survival on the Titanic dataset. This project provides a complete ML pipeline with preprocessing, model serving and testing.

## 🚀 Features

- **FastAPI REST API** with automatic documentation
- **Machine Learning Model** for survival prediction
- **Data Preprocessing Pipeline** with feature engineering
- **Testing** (unit, integration, end-to-end)
- **Prometheus Metrics** for monitoring
- **Docker Support** for containerization

## 📁 Project Structure

```
titanic/
├── src/                          # Source code
│   ├── main.py                   # FastAPI application
│   ├── preprocess_data.py        # Data preprocessing pipeline
│   └── models/                   # Trained model files
├── tests/                        # Test suite
│   ├── conftest.py               # Shared test fixtures
│   ├── unit/                     # Unit tests
│   │   ├── test_main.py          # API endpoint tests
│   │   ├── test_models.py        # Pydantic model tests
│   │   ├── test_preprocess_data.py # Preprocessing tests
│   │   └── test_helpers.py       # Utility function tests
│   └── integration/              # Integration tests
│       ├── test_api_workflows.py # API workflow tests
│       ├── test_end_to_end.py    # End-to-end tests
│       └── test_preprocessing_pipeline.py # Pipeline tests
├── data/                         # Dataset files
├── notebooks/                    # Jupyter notebooks
├── docs/                         # Documentation
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker configuration
├── pytest.ini                   # Pytest configuration
└── .gitignore                   # Git ignore rules
```

## 🛠️ Installation

### Prerequisites

- Python 3.9
- pip or conda

### Setup

1. **Clone the repository:**

2. **Create and activate virtual environment:**
   ```bash
   python3.9 -m venv titanic
   source titanic/bin/activate  # On Windows: titanic\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🐳 Docker

### Building the Image

```bash
docker build -t titanic-api .
```

### Running with Docker

```bash
docker run -p 8000:8000 titanic-api
```

### Making Predictions

**Example API call:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
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
     }'
```

**Response:**
```json
{
  "prediction": 0,
  "prediction_label": "Not Survived",
  "probabilities": [0.7, 0.3],
  "message": "Prediction completed successfully."
}
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/           # Unit tests only
pytest tests/integration/    # Integration tests only
```

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | API health check |
| `/predict` | POST | Make survival prediction |
| `/load` | POST | Load new model |
| `/history` | GET | Get prediction history |
| `/metrics` | GET | Prometheus metrics |
| `/docs` | GET | API documentation |

## 📈 Monitoring

### Prometheus Metrics

The API exposes comprehensive Prometheus metrics for monitoring and alerting:

- **Request Metrics**: Total requests
- **Error Metrics**: Categorized errors with labels for error type and status code
- **State Metrics**: Prediction history count

#### Key Metrics:
- `titanic_prediction_requests_total` - Total prediction requests
- `titanic_prediction_errors_total` - Errors with labels for categorization (`error_type`, `status_code`)

#### Error Categories:
- `validation_error` - Data validation errors
- `internal_error` - Unexpected server errors
- `model_not_loaded` - Model not available
- `model_not_found` - Model file not found
- `model_load_error` - Model loading failures

## 📝 License

This project is licensed under the MIT License.