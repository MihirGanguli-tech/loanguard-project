from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
import pandas as pd


class DropColumns(BaseEstimator, TransformerMixin):
    '''
    Drop columns with high proportion of missing values, as well as other columns that were decided to be dropped in EDA.
    '''

    def __init__(self, prop_missing_threshold = 0.474):
        self.prop_missing_threshold = prop_missing_threshold
        
    def fit(self, X, y = None):

        #find columns with high missingness, above prop_missing_threshold, add to list to be dropped
        self.cols_to_drop = [col for col in X.columns if X[col].isna().mean() > self.prop_missing_threshold and col not in ['EXT_SOURCE_1', 'OWN_CAR_AGE']]
        self.cols_to_drop.extend(['FLAG_OWN_CAR', 'WEEKDAY_APPR_PROCESS_START', 'REGION_RATING_CLIENT', 'REG_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_WORK_CITY', 'SK_ID_CURR'])
        return self
    
    def transform(self, X, y = None):
        df = X.copy()
        df = df.drop(columns = self.cols_to_drop, errors = 'ignore')

        return df

class DaysEmployedFixer(BaseEstimator, TransformerMixin):
    '''Replace anomaly value 365243 (representing unemployment/retirement) with nan'''
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        df = X.copy()
        df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)

        return df

        



class DaysColumnsTransformer(BaseEstimator, TransformerMixin):
    ''' 
    Convert DAYS_BIRTH column to AGE_YEARS.
    Convert rest of DAYS columns to positive values.
    Drop original DAYS_BIRTH column.
     '''

    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):

        df = X.copy()
    
        df['AGE_YEARS'] = (-pd.to_numeric(df['DAYS_BIRTH'], errors='coerce') / 365)

        days_cols = [col for col in df.columns if col.startswith('DAYS') and col != 'DAYS_BIRTH']
        for col in days_cols:
            df[col] = -pd.to_numeric(df[col], errors='coerce')

        df = df.drop(columns='DAYS_BIRTH', errors='ignore')
        return df

class FlagEngineer(BaseEstimator, TransformerMixin):
    '''Create flag for missing/not missing for columns where missingness itself is an indicator, 
    and cannot be imputed without adding extra assumptions.
    '''
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        df = X.copy()

        df['EXT_SOURCE_1_MISSING_FLAG'] = df['EXT_SOURCE_1'].isna().astype(int)
        df['EXT_SOURCE_3_MISSING_FLAG'] = df['EXT_SOURCE_3'].isna().astype(int)
        df['DAYS_EMPLOYED_MISSING_FLAG'] = df['DAYS_EMPLOYED'].isna().astype(int)
        
        return df


        

class RatioEngineer(BaseEstimator, TransformerMixin):
    '''
    Create credit to goods ratio column and annuity to income ratio column, engineered features to avoid collinearity. 
    '''
    
    def fit (self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        df = X.copy()

        df['CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT']/df['AMT_GOODS_PRICE']
        df['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY']/df['AMT_INCOME_TOTAL']

        df = df.drop(columns= ['AMT_GOODS_PRICE', 'AMT_ANNUITY'], errors= 'ignore')

        return df

class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    '''Group infrequent categories into "Other"
    learns rare categories from training data only to prevent leakage.
    '''
    
    def __init__(self, cols, threshold=0.03):
        #threshold for value to be considered rare category
        self.cols = cols
        self.threshold = threshold
    
    def fit(self, X, y=None):
        self.rare_categories_ = {}
        
        for col in self.cols:
            # calculate frequency of each category
            freq = X[col].value_counts(normalize=True)
            # store categories below threshold
            self.rare_categories_[col] = list(freq[freq < self.threshold].index)
        
        return self
    
    def transform(self, X, y=None):
        df = X.copy()
        
        for col in self.cols:
            # replace rare categories with 'Other'
            df[col] = df[col].apply(
                lambda x: 'Other' if x in self.rare_categories_[col] else x
            )
        
        return df
    
def build_pipeline(num_cols, cat_cols, model, scale_features = False):
    
    #median imputation as many columns are skewed right, scaling included if necessary for model
    if scale_features == True:
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])
    else:
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median'))])
    
    # categorical pipeline - mode imputation and then one hot encoding
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # apply num and cat tranformations for respective columns
    col_transformer = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    
    pipeline = Pipeline([
        ('drop_columns', DropColumns()),
        ('days_employed_fixer', DaysEmployedFixer()),
        ('flag_engineer', FlagEngineer()),
        ('days_transformer', DaysColumnsTransformer()),
        ('ratio_engineer', RatioEngineer()),
        ('rare_category_grouper', RareCategoryGrouper(cols=['OCCUPATION_TYPE', 'ORGANIZATION_TYPE'])),
        ('col_transformer', col_transformer),
        ('model', model)
    ])
    
    return pipeline