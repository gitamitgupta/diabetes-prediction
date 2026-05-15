# src/train.py
import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import tf2onnx
import onnx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

def get_raw_dataset():
    """Loads the dataset directly from the local root directory or data folder."""
    if os.path.exists("diabetes.csv"):
        print("[INFO] Loading 'diabetes.csv' from root directory...")
        return pd.read_csv("diabetes.csv")
    elif os.path.exists("data/diabetes.csv"):
        print("[INFO] Loading 'diabetes.csv' from data/ directory...")
        return pd.read_csv("data/diabetes.csv")
    else:
        raise FileNotFoundError(
            "🚨 Critical Error: Could not find 'diabetes.csv'. "
            "Please ensure the file is placed in your project root directory."
        )

def main():
    ONNX_OUT = 'models/diabetes_ann_model.onnx'
    SCALER_OUT = 'models/scaler.pkl'
    TEMP_SAVED_MODEL = 'models/temp_saved_model'
    os.makedirs('models', exist_ok=True)

    # 1. Load Data
    df = get_raw_dataset()

    # Data Imputation: Replace invalid 0s with median in physiological columns
    print("[INFO] Preprocessing data and cleaning invalid biological zero metrics...")
    impute_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in impute_cols:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())

    # Split targets and features
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 2. Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_OUT)
    print(f"[INFO] Preprocessing Scaler exported cleanly to {SCALER_OUT}")

    # 3. Model Architecture Topology
    model = Sequential([
        Input(shape=(X_train_scaled.shape[1],), name='input_layer'),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 4. Neural Network Training
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    print("[INFO] Executing Keras training cycles...")
    model.fit(
        X_train_scaled, y_train,
        epochs=120,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1 
    )

    # Evaluate performance metrics
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"\n[INFO] Baselined Test Evaluation Accuracy: {accuracy * 100:.2f}%")

    # Display clean classification metrics matrix
    y_preds = (model.predict(X_test_scaled) > 0.5).astype(int)
    print("\nDetailed Classification Metrics Report:")
    print(classification_report(y_test, y_preds))

    # 5. ONNX Binary Compilation via Modern Keras Export Path
    print("\n[INFO] Exporting SavedModel directory format via modern Keras API...")
    # This uses model.export() instead of model.save() to compile standard TF serving files
    model.export(TEMP_SAVED_MODEL)

    print("[INFO] Converting SavedModel directory into lightweight ONNX format...")
    # Use the directory-to-onnx conversion interface command
    os.system(f"python -m tf2onnx.convert --saved-model {TEMP_SAVED_MODEL} --output {ONNX_OUT} --opset 13")
    
    if os.path.exists(ONNX_OUT):
        print(f"[SUCCESS] Deep Learning Architecture safely serialized to {ONNX_OUT}")
        
        # Clean up the intermediate directory to keep your repository tidy
        import shutil
        if os.path.exists(TEMP_SAVED_MODEL):
            shutil.rmtree(TEMP_SAVED_MODEL)
            print("[INFO] Intermediate directory cleanup completed successfully.")
    else:
        print("🚨 Error: ONNX file conversion failed.")

if __name__ == '__main__':
    main()