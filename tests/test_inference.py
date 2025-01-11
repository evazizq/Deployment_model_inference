import pytest
import pandas as pd
import numpy as np
import joblib
from src.inference import main
from src.model_loader import load_model
from src.data_preprocessor import preprocessor_data

class MockModel:
    def predict(self, X):
        return np.array([1])

@pytest.fixture
def mock_model(tmp_path):
    """Create a mock model that returns predetermined predictions."""
    model = MockModel()
    model_path = tmp_path / "test_model.joblib"
    joblib.dump(model, model_path)
    return model_path

@pytest.fixture
def sample_input_data():
    """Provide sample input data for testing."""
    return {
        "age": 67,
        "sex": 1,
        "cp": 4,
        "trestbps": 120,
        "chol": 237,
        "fbs": 0,
        "restecg": 0,
        "thalach": 71,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 2,
        "ca": 3,
        "thal": 2
    }

@pytest.fixture
def columns_to_impute():
    """Provide columns to impute for testing."""
    return ["age", "trestbps", "chol", "thalach", "oldpeak"]

def test_end_to_end_inference(mock_model, sample_input_data, columns_to_impute):
    """Test the end-to-end inference process."""
    model = load_model(mock_model)
    preprocessed_data = preprocessor_data(sample_input_data, columns_to_impute)
    prediction = model.predict(preprocessed_data)
    assert prediction[0] in [0, 1]

def test_inference_with_edge_cases(mock_model, columns_to_impute):
    """Test inference with edge cases."""
    model = load_model(mock_model)
    edge_case_data = {
        "age": 120,
        "sex": 1,
        "cp": 4,
        "trestbps": 300,
        "chol": 600,
        "fbs": 1,
        "restecg": 2,
        "thalach": 30,
        "exang": 1,
        "oldpeak": 10.0,
        "slope": 3,
        "ca": 4,
        "thal": 3
    }
    preprocessed_data = preprocessor_data(edge_case_data, columns_to_impute)
    prediction = model.predict(preprocessed_data)
    assert prediction[0] in [0, 1]