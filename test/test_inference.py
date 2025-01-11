import pytest
from src.inference import run_inference

def test_run_inference(mocker):
    # Mock the load_model and preprocessor_data functions
    mocker.patch('src.inference.load_model')
    mocker.patch('src.inference.preprocessor_data', return_value=[[125, 212, 168, 1.0]])
    mock_model = mocker.Mock()
    mock_model.predict.return_value = [0]
    src.inference.load_model.return_value = mock_model

    input_data = {
        "age": 52,
        "sex": 1,
        "cp": 0,
        "trestbps": 125,
        "chol": 212,
        "fbs": 0,
        "restecg": 1,
        "thalach": 168,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 2,
        "ca": 2,
        "thal": 3,
    }
    model_path = "models/trained_model_2025-01-10.joblib"
    prediction = run_inference(model_path, input_data)
    assert prediction == [0]