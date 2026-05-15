# src/evaluate.py
import os
import joblib
import numpy as np
import pandas as pd
import onnxruntime as ort
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

# Reuse the exact same preprocessing split logic to match your training set
if os.path.exists("diabetes.csv"):
    df = pd.read_csv("diabetes.csv")
else:
    raise FileNotFoundError("Could not find diabetes.csv in root.")

# Clean invalid biological zeros
impute_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in impute_cols:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

X = df.drop(columns=['Outcome'])
y = df['Outcome']
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Load production assets
session = ort.InferenceSession('models/diabetes_ann_model.onnx')
scaler = joblib.load('models/scaler.pkl')

# Scale validation data
X_test_scaled = scaler.transform(X_test).astype(np.float32)

# Run Inference via ONNX Engine
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
y_pred_probs = session.run([output_name], {input_name: X_test_scaled})[0].flatten()
y_pred_classes = (y_pred_probs > 0.5).astype(int)

# -----------------------------------------------------------------
# 1. GENERATE VISUAL CONFUSION MATRIX
# -----------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Non-Diabetic', 'Predicted Diabetic'],
            yticklabels=['Actual Non-Diabetic', 'Actual Diabetic'])
plt.title('ANN Confusion Matrix Evaluation')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('models/confusion_matrix.png', dpi=300)
plt.close()
print("[SUCCESS] Confusion matrix visualization saved to 'models/confusion_matrix.png'")

# -----------------------------------------------------------------
# 2. GENERATE ROC CURVE MATRIX
# -----------------------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (Area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('models/roc_curve.png', dpi=300)
plt.close()
print("[SUCCESS] ROC Curve metrics saved to 'models/roc_curve.png'")