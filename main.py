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
autoencoder = tf.keras.models.load_model(os.getenv("MODEL_PATH_AUTOENCODER", "models/autoencoder_model.h5"), custom_objects={"mse": MeanSquaredError()})

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

    # Advanced Feature Engineering
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)

    power_transformer = PowerTransformer()
    X_test_transformed = power_transformer.fit_transform(X_test_scaled)

    X_test_transformed_np = np.array(X_test_transformed)

    # Predict anomalies
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
    df["Anomaly_Prediction"] = ensemble_predictions

    # Classify anomaly types
    anomaly_types = []
    for index, row in df.iterrows():
        if row["Anomaly_Prediction"] == 1:
            if row["Transaction_Amount"] > df["Transaction_Amount"].quantile(0.99):
                anomaly_types.append("High Transaction Amount")
            elif row["Return_Frequency"] > df["Return_Frequency"].quantile(0.99):
                anomaly_types.append("High Return Frequency")
            elif row["Transaction_Velocity"] > df["Transaction_Velocity"].quantile(0.99):
                anomaly_types.append("Rapid Transactions")
            elif row["Location_Distance"] > df["Location_Distance"].quantile(0.99):
                anomaly_types.append("Suspicious Location")
            elif row["High_Risk_Payment"] == 1:
                anomaly_types.append("High-Risk Payment Method")
            elif row["High_Return_User"] == 1:
                anomaly_types.append("Frequent Return User")
            else:
                anomaly_types.append("General Anomaly")
        else:
            anomaly_types.append("Normal")
    
    df["Anomaly_Type"] = anomaly_types

    # Save results as CSV
    output_file = "ensemble_anomaly_predictions.csv"
    df.to_csv(output_file, index=False)

    return {"predictions": df.to_dict(orient="records")}

# API to download the prediction results
@app.get("/download/")
async def download_results():
    file_path = "ensemble_anomaly_predictions.csv"
    return FileResponse(path=file_path, filename="anomaly_predictions.csv", media_type="text/csv")
