import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from sklearn.preprocessing import StandardScaler, PowerTransformer
import uvicorn
from tensorflow.keras.losses import MeanSquaredError
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://anomaly-detection-in-e-commerce.netlify.app"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models from environment variables
random_forest_model = joblib.load(os.getenv("MODEL_PATH_RF", "models/random_forest_model.pkl"))
xgb_model = joblib.load(os.getenv("MODEL_PATH_XGB", "models/xgboost_model.pkl"))
autoencoder = tf.keras.models.load_model(os.getenv("MODEL_PATH_AUTOENCODER", "models/autoencoder_model.h5"),custom_objects={"mse": MeanSquaredError()})

@app.get("/")
def home():
    return {"message": "Anomaly Detection API Running Successfully"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

# Upload & Predict API
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    feature_columns = ["Transaction_Amount", "Return_Frequency", "Latitude", "Longitude",
                       "Transaction_Velocity", "High_Risk_Payment", "High_Return_User", "Location_Distance"]
    
    X_test = df[feature_columns]

    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)

    power_transformer = PowerTransformer()
    X_test_transformed = power_transformer.fit_transform(X_test_scaled)

    # Predict anomalies
    X_test_pred = autoencoder.predict(X_test_transformed)
    reconstruction_error = np.mean(np.abs(X_test_transformed - X_test_pred), axis=1)

    mean_error = np.mean(reconstruction_error)
    std_error = np.std(reconstruction_error)
    threshold = mean_error + 1.5 * std_error

    y_pred_autoencoder = (reconstruction_error > threshold).astype(int)

    # Ensemble method
    y_pred_rf = random_forest_model.predict(X_test_transformed)
    y_pred_xgb = xgb_model.predict(X_test_transformed)
    ensemble_predictions = ((y_pred_rf + y_pred_xgb + y_pred_autoencoder) >= 2).astype(int)

    df["Anomaly_Prediction"] = ensemble_predictions

    # Save results as CSV
    output_file = "ensemble_anomaly_predictions.csv"
    df.to_csv(output_file, index=False)

    return {"predictions": df.to_dict(orient="records")}

# API to download the prediction results
@app.get("/download/")
async def download_results():
    file_path = "ensemble_anomaly_predictions.csv"
    return FileResponse(path=file_path, filename="anomaly_predictions.csv", media_type="text/csv")
