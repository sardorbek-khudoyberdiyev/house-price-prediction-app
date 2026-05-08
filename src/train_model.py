import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def main():
    #Load dataset
    df = pd.read_csv('data/raw/Housing.csv')

    #Select features and target variable
    features = [
        "area",
        "bedrooms",
        "bathrooms",
        "stories",
        "mainroad",
        "guestroom",
        "basement",
        "hotwaterheating",
        "airconditioning",
        "parking",
        "prefarea",
        "furnishingstatus"
    ]
    target = "price"

    X = df[features]
    y = df[target]

    #Convert categorical variables to dummy variables
    X = pd.get_dummies(X, drop_first=True)

    #Save features names used for training
    features_names = list(X.columns)

    #Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size=0.2, 
                                                        random_state=42)
    #Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    #Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    print(f"Mean Absolute Error: {mae}")
    print(f"Root Mean Squared Error: {rmse}")

    #Save the model and features names
    joblib.dump(model, 'models/house_price_model.pkl')
    joblib.dump(features_names, 'models/features_names.pkl')

    print("\nModel and features names saved successfully.")

if __name__ == "__main__":
    main()