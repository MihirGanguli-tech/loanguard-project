# LoanGuard — Credit Default Prediction

**[Live API →](https://whawj3m6rd.execute-api.us-east-1.amazonaws.com/docs#/default/predict_default_predict_post)**



An end to end machine learning project that predicts the likelihood of a loan applicant defaulting, built on the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset from Kaggle.

The project covers the full ML workflow: Exploratory Data Analysis, feature engineering, preprocessing pipeline, model training and evaluation, and a REST API for serving predictions.

## Sample Requests
Three sample JSON payloads (low, medium, and high risk applicants) are available in [`tests/sample_requests.json`](tests/sample_requests.json) for testing the live API.

EDA was originally done for a class project, expanded as a personal project and resume builder.

---

## Tech Stack

- **Python** — core language
- **Pandas, NumPy** — data manipulation
- **Scikit-learn** — preprocessing pipeline and model training
- **LightGBM** — gradient boosting model, chosen after testing against Logistic Regression and Random Forest
- **FastAPI** — REST API
- **Uvicorn** — ASGI server
- **Joblib** — model serialization


---

## Project Structure
```
loanguard-project/
├── app/
│   ├── main.py          # FastAPI app and endpoints
│   ├── predictor.py     # Model loading and prediction logic
│   └── schemas.py       # Pydantic request/response schemas
├── data/
│   ├── raw/             # Raw dataset (not tracked in git)
│   └── processed/       # Train/test splits (not tracked in git)
├── models/              # Saved model files (not tracked in git)
├── notebooks/
│   └── EDA.ipynb        # Exploratory Data Analysis
├── results/
│   └── model_results.ipynb  # Model comparison and results, tracked after running different models and tuning
├── src/
│   ├── preprocess.py    # Custom sklearn transformers and pipeline, prevent data leakage
│   ├── train.py         # Model training and validation
│   └── evaluate.py      # Final test set evaluation
├── tests/
│   └── test_api.py      # API endpoint tests
├── Procfile
├── pyproject.toml
└── requirements.txt
```

---

## Model Results

Target variable is heavily imbalanced (~92% non-default, 8% default). Models evaluated using ROC-AUC and Recall for the default class.

| Model | ROC-AUC | Recall (class 1) |
|-------|---------|-----------------|
| LightGBM (default) | 0.754 | 0.67 |
| Logistic Regression | 0.741 | 0.68 |
| Random Forest | 0.749 | 0.56 |

**Final model:** LightGBM with default parameters  
**Test set ROC-AUC:** 0.753

---

## Preprocessing Pipeline

Built as a custom sklearn pipeline with the following steps:

- Drop high missingness columns and collinear features
- Fix `DAYS_EMPLOYED` anomaly (sentinel value 365243 → NaN)
- Engineer missingness flags for `EXT_SOURCE_1`, `EXT_SOURCE_3`, and `DAYS_EMPLOYED`
- Convert `DAYS_BIRTH` to `AGE_YEARS`, convert remaining DAYS columns to positive
- Engineer ratio features: `CREDIT_TO_GOODS_RATIO`, `ANNUITY_TO_INCOME_RATIO`
- Group rare categories in `OCCUPATION_TYPE` and `ORGANIZATION_TYPE`
- Median imputation for numerical columns (most continuous columns skewed right), mode imputation for categorical
- One-hot encoding for categorical columns

---

## Architecture

The trained LightGBM model is served through a FastAPI application, containerized with Docker, and deployed as an AWS Lambda container behind an API Gateway HTTP API, which  provides a fully serverless endpoint.

```text
Client
   │
   ▼
API Gateway (HTTP API)
   │
   ▼
AWS Lambda (Docker Container)
   │
   ▼
FastAPI (via Mangum)
   │
   ▼
LightGBM Model
```

## Why I chose Lambda + API Gateway over EC2

The project was initially deployed on AWS Elastic Beanstalk, but was draining my free credits at around $8 per month. For the production deployment linked in this repository, I chose **AWS Lambda + API Gateway** instead, better fit for a portfolio project.

- **Near $0/month hosting:** Unlike an EC2 instance, which incurs charges while running (~$7–8/month for a `t3.micro`), Lambda's permanent free tier means this API operates at  $0/month since the project is only deployed as a portfolio project.

- **Serverless:** No need to provision or manage servers

- **Pay per request:** The API incurs no compute cost when idle because the billing is based on usage (requests and compute time).  API Gateway throttling set to 5 requests per second in the very unlikely event that there is abuse or extreme traffic spikes.
---

## Challenges 

- Package size exceeded Lambda's 250MB limit

Initially tried to upload a zip file through the console and then through s3 when the zip file exceeded the 50 mb limit. Some required dependencies were also missing from the initial requirements file, I had created a new requirements file so that unnecessary packages were not added. Once all dependencies were added, the unzipped package exceeded Lambda's 250MB limit.

 Solution: Migrated to a Docker container image deployment which has a limit of 10 GB using AWS's official public.ecr.aws/lambda/python:3.10 base image.

- Model file not found at runtime despite being present in the image

Throughout the process I was looking at Cloudwatch logs to identify any errors.  Even after confirming the model file was correctly copied into the container at build time, the Lambda function threw FileNotFoundError on a relative path (models/lightgbm.joblib).

 Solution: Lambda's execution environment doesn't guarantee the working directory matches the code's location. Resolved by loading the model using an absolute path derived from the module's own file location (os.path.dirname(os.path.abspath(__file__))), matching the pattern already used in the preprocess.py.





---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Predict default risk for a loan applicant |

**Example request:**
```json
{
    "AMT_CREDIT": 500000,
    "AMT_INCOME_TOTAL": 150000,
    "AMT_ANNUITY": 25000,
    "AMT_GOODS_PRICE": 450000,
    "DAYS_BIRTH": -12000,
    "DAYS_EMPLOYED": -2000,
    "EXT_SOURCE_1": 0.5,
    "EXT_SOURCE_2": 0.6,
    "EXT_SOURCE_3": 0.7,
    "NAME_CONTRACT_TYPE": "Cash loans",
    "CODE_GENDER": "M"
}
```

**Example response:**
```json
{
    "prediction": 0,
    "probability": 0.34,
    "risk_level": "Medium"
}
```

---

## Possible Future Improvements

- Log transform for skewed numerical features before scaling for linear models
- pytorch neural network model (most likely would not outperform lightGBM anyway, due to high proportion of missing values and mixed data types)
- Error handling to ensure that data entered for the predictions is in the correct format, with correct type of values.

