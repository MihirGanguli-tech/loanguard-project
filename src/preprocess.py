from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class DropColumns(BaseEstimator, TransformerMixin):
    '''
    Drop columns with high proportion of missing values, as well as other columns that were decided to be dropped in EDA.
    '''

    def __init__(self, prop_missing_threshold):
        self.prop_missing_threshold = prop_missing_threshold
        
    def fit(self, X, y = None):

        #find columns with high missingness, above prop_missing_threshold, add to list to be dropped
        self.cols_to_drop = [col for col in X.columns if X[col].isna().mean() > self.prop_missing_threshold and col not in ['EXT_SOURCE_1', 'OWN_CAR_AGE']]
        self.cols_to_drop.extend(['FLAG_OWN_CAR', 'WEEKDAY_APPR_PROCESS_START'])
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
        #convert to days since birth age in years, easier to interpret
        df['AGE_YEARS'] = (-df['DAYS_BIRTH'] / 365).astype(int)

        days_cols = [col for col in df.columns if col.startswith('DAYS')]
        for col in days_cols:
            df[col] = -df[col]

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
    pass

class CategoryGrouper(BaseEstimator, TransformerMixin):
    pass

def build_pipeline():
    
    pass