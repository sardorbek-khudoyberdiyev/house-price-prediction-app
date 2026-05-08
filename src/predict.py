import pandas as pd
import joblib

MODEL_PATH = 'models/house_price_model.pkl'
FEATURES_PATH = 'models/features_names.pkl'

def load_model():
    model = joblib.load(MODEL_PATH)
    features_names = joblib.load(FEATURES_PATH)
    return model, features_names

def predict(input_data):
    model, features_names = load_model()

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])   

    input_encoded = pd.get_dummies(input_df, drop_first=True)

    input_encoded = input_encoded.reindex(columns=features_names, fill_value=0)

    prediction = model.predict(input_encoded)[0]
    return prediction