# LoanGuard — Credit Default Prediction

**[Live API →](http://loanguard-env.eba-vmmptjpn.us-east-1.elasticbeanstalk.com/docs)**

-Test cases in tests/sample_requests.json

An end to end machine learning project that predicts the likelihood of a loan applicant defaulting, built on the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset from Kaggle.

The project covers the full ML workflow: Exploratory Data Analysis, feature engineering, preprocessing pipeline, model training and evaluation, and a REST API for serving predictions.

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
- **Docker** — containerization

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
├── Dockerfile
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
**Test set ROC-AUC:** 0.743

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

## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/yourusername/loanguard-project.git
cd loanguard-project
```

**2. Create and activate a conda environment**
```bash
conda create -n loanguard python=3.10
conda activate loanguard
```

**3. Install dependencies**
```bash
pip install -e .
pip install -r requirements.txt
```

**4. Add the dataset**  
Download `application_train.csv` from [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data) and place it in `data/raw/`.

**5. Train the model**
```bash
python -m src.train
```

**6. Run the API**
```bash
uvicorn app.main:app --reload
```

**7. Visit the interactive docs**  
Open `http://127.0.0.1:8000/docs` in your browser.

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

## Future Improvements

- Log transform for skewed numerical features before scaling for linear models
- pytorch neural network model (most likely would not outperform lightGBM anyway, due to high proportion of missing values and mixed data types)
- containerize with Docker
- Streamlit frontend
- CI/CD pipeline with GitHub Actions

