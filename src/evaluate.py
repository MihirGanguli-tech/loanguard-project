import os
import joblib
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, recall_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#load the saved lightgbm model with default hyperparams
pipeline = joblib.load(os.path.join(BASE_DIR, "models", "lightgbm.joblib"))

#load saved X_test and y_test
X_test = joblib.load(os.path.join(BASE_DIR, "data", "processed", "X_test.joblib"))
y_test = joblib.load(os.path.join(BASE_DIR, "data", "processed", "y_test.joblib"))


y_test_pred = pipeline.predict(X_test)
y_test_proba = pipeline.predict_proba(X_test)[:, 1]
    
roc_auc = roc_auc_score(y_test, y_test_proba)

print(f"ROC-AUC: {roc_auc:.4f}")
print(classification_report(y_test, y_test_pred))