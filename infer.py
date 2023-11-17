import pandas as pd
import joblib

def load_model(model_path):
    # Load your pre-trained model
    model = joblib.load(model_path)
    return model

def infer(model, data):
    # Perform inference using the loaded model
    predictions = model.predict(data)
    return predictions

if __name__ == "__main__":
    # Example usage
    model_path = "../models/your_model.joblib"
    data_path = "../data/new_data.csv"

    model = load_model(model_path)

    # Load new data
    new_data = pd.read_csv(data_path)

    # Perform inference
    results = infer(model, new_data)

    # Print or save results as needed
    print(results)
