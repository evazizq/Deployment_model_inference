import sys
import os

# Agregar la raíz del proyecto al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model_loader import load_model
from src.data_preprocessor import preprocessor_data

def main():

    # Load the model
    model_path = "models/trained_model_2025-01-10.joblib"
    model = load_model(model_path)
    print("paso la carga del modelo")

    # Sample prediction
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
    "oldpeak": 1,
    "slope": 2,
    "ca": 2,
    "thal": 3,
}

    # columns to use
    columns_to_use = ['trestbps', 'chol', 'thalach', 'oldpeak']

    # reply preprocessor of the training 
    preprocessed_data = preprocessor_data(data=input_data, columns_to_impute=columns_to_use)
    print("Procesamiento realizado con exito")

    try: 
        prediction = model.predict(preprocessed_data)
        print(f"Predictions: {prediction}")
    except Exception as e:
        print(f"Error running inference: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()