import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.preprocess import build_pipeline, DropColumns, DaysEmployedFixer, DaysColumnsTransformer, FlagEngineer, RatioEngineer, RareCategoryGrouper
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score, recall_score
import joblib

df = pd.read_csv("data/raw/application_train.csv")

X = df.drop(columns = 'TARGET')
y= df['TARGET']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=23, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=23, stratify=y_temp)

joblib.dump(X_test, 'data/processed/X_test.joblib')
joblib.dump(y_test, 'data/processed/y_test.joblib')
print("test splits saved")

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




models = {
    'logistic_regression': LogisticRegression(
        class_weight='balanced', 
        max_iter=1000, 
        random_state=23),
    'random_forest': RandomForestClassifier(
        max_depth=10,
        min_samples_leaf=50,
        class_weight='balanced',
        random_state=23,
        n_jobs=-1),
    'lightgbm': LGBMClassifier(
        class_weight='balanced',
        random_state=23,
        n_jobs=-1
    )}

results = {}

for model_name, model in models.items():
    print(f"{model_name}")
    
    # build pipeline with correct scale_features for linear models
    if model_name == 'logistic_regression':
        scale = True 
    else:
        scale = False
    
    pipeline = build_pipeline(num_cols, cat_cols, model, scale_features=scale)
    pipeline.fit(X_train, y_train)
    
    y_val_pred = pipeline.predict(X_val)
    y_val_proba = pipeline.predict_proba(X_val)[:, 1]
    
    roc_auc = roc_auc_score(y_val, y_val_proba)
    
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(classification_report(y_val, y_val_pred))
    
    # save each fitted pipeline
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, f'models/{model_name}.joblib')
    
    # store results for comparison
    results[model_name] = roc_auc

# print summary
print("\n=== Model Comparison (ROC-AUC) ===")
for model_name, roc_auc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{model_name}: {roc_auc:.4f}")

#param dist for randomized search
param_dist = {
    'model__n_estimators': [100, 300, 500],
    'model__max_depth': [3, 5, 7],        
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__num_leaves': [31, 63, 127],
    'model__min_child_samples': [20, 50, 100]
}

lgbm_pipeline = build_pipeline(
    num_cols, 
    cat_cols, 
    LGBMClassifier(class_weight='balanced', random_state=23, n_jobs=-1),
    scale_features=False
)

search = RandomizedSearchCV(
    lgbm_pipeline,
    param_distributions=param_dist,
    n_iter=20,              # try 20 random combinations
    scoring='roc_auc',      # optimize for ROC-AUC
    cv=3,                   # 3 fold cross validation
    random_state=23,
    n_jobs=-1,              # use all available cpu cores
    verbose=2               # prints progress
)

search.fit(X_train, y_train)

print(f"Best params: {search.best_params_}")
print(f"Best CV ROC-AUC: {search.best_score_:.4f}")

# evaluate best model on validation set
y_val_pred = search.predict(X_val)
y_val_proba = search.predict_proba(X_val)[:, 1]
print(f"Validation ROC-AUC: {roc_auc_score(y_val, y_val_proba):.4f}")
print(classification_report(y_val, y_val_pred))

# save best pipeline
joblib.dump(search.best_estimator_, 'models/lightgbm_tuned.joblib')