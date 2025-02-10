from fastapi import FastAPI, File, UploadFile
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import io
from sklearn.preprocessing import StandardScaler, PowerTransformer

app = FastAPI()

# Load models
print("Loading models...")
random_forest_model = joblib.load("models/random_forest_model.pkl")
xgb_model = joblib.load("models/xgboost_model.pkl")
autoencoder = tf.keras.models.load_model("models/autoencoder_model.keras")
print("Models loaded successfully!")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read CSV file
        dataset = pd.read_csv(io.StringIO(file.file.read().decode('utf-8')))
        
        # Define feature columns
        feature_columns = ["Transaction_Amount", "Return_Frequency", "Latitude", "Longitude",
                           "Transaction_Velocity", "High_Risk_Payment", "High_Return_User", "Location_Distance"]
        X_test = dataset[feature_columns]
        
        # Preprocessing
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)
        
        power_transformer = PowerTransformer()
        X_test_transformed = power_transformer.fit_transform(X_test_scaled)
        
        X_test_transformed_np = np.array(X_test_transformed)
        
        # Model Predictions
        y_pred_rf = random_forest_model.predict(X_test_transformed_np)
        y_pred_xgb = xgb_model.predict(X_test_transformed_np)
        
        X_test_pred = autoencoder.predict(X_test_transformed_np)
        reconstruction_error = np.mean(np.abs(X_test_transformed_np - X_test_pred), axis=1)
        
        mean_error = np.mean(reconstruction_error)
        std_error = np.std(reconstruction_error)
        threshold = mean_error + 1.5 * std_error
        
        y_pred_autoencoder = (reconstruction_error > threshold).astype(int)
        
        # Ensemble prediction
        ensemble_predictions = ((y_pred_rf + y_pred_xgb + y_pred_autoencoder) >= 2).astype(int)
        dataset["Anomaly_Prediction"] = ensemble_predictions
        
        return dataset.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def home():
    return {"message": "Ensemble anomaly detection API is running!"}
