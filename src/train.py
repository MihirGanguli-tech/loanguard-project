import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.preprocess import build_pipeline, DropColumns, DaysEmployedFixer, DaysColumnsTransformer, FlagEngineer, RatioEngineer, RareCategoryGrouper
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

df = pd.read_csv("data/raw/application_train.csv")

X = df.drop(columns = 'TARGET')
y= df['TARGET']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=23, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=23, stratify=y_temp)

early_pipeline = Pipeline([
    ('drop_columns', DropColumns()),
    ('days_employed_fixer', DaysEmployedFixer()),
    ('flag_engineer', FlagEngineer()),
    ('days_transformer', DaysColumnsTransformer()),
    ('ratio_engineer', RatioEngineer()),
    ('rare_category_grouper', RareCategoryGrouper(cols=['OCCUPATION_TYPE', 'ORGANIZATION_TYPE'])),
])

X_train_transformed = early_pipeline.fit_transform(X_train)

# now derive correct column lists
num_cols = list(X_train_transformed.select_dtypes(include='number').columns)
cat_cols = list(X_train_transformed.select_dtypes(include='object').columns)

pipeline = build_pipeline(num_cols, cat_cols, LogisticRegression(class_weight='balanced', max_iter=1000), scale_features=True)


pipeline.fit(X_train, y_train)


y_val_pred = pipeline.predict(X_val)

y_val_proba = pipeline.predict_proba(X_val)[:,1]
print(f"ROC-AUC for Logistic Regression: {roc_auc_score(y_val, y_val_proba):.4f}")


print(classification_report(y_val, y_val_pred))


print("reached save section")
joblib.dump(pipeline, 'models/baseline_logistic_regression.joblib')
print("pipeline saved")
joblib.dump(X_test, 'data/processed/X_test.joblib')
joblib.dump(y_test, 'data/processed/y_test.joblib')
print("test splits saved")

rf_pipeline = build_pipeline(num_cols, cat_cols, 
                             RandomForestClassifier(class_weight='balanced',
                                                     max_depth = 20, #prevent overfitting from trees being too deep
                                                     min_samples_leaf=50,
                                                     n_jobs = -1) #using all cpu cores to speed up training
                            )


rf_pipeline.fit(X_train, y_train)


y_val_pred = rf_pipeline.predict(X_val)

y_val_proba = rf_pipeline.predict_proba(X_val)[:,1]
print(f"ROC-AUC for Random Forest: {roc_auc_score(y_val, y_val_proba):.4f}")


print(classification_report(y_val, y_val_pred))


print("reached save section")
joblib.dump(rf_pipeline, 'models/baseline_random_forest.joblib')
print(" random forest pipeline saved")
joblib.dump(X_test, 'data/processed/X_test.joblib')
joblib.dump(y_test, 'data/processed/y_test.joblib')
print("test splits saved")
