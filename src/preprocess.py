from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

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
    pass

class RatioEngineer(BaseEstimator, TransformerMixin):
    pass

class CategoryGrouper(BaseEstimator, TransformerMixin):
    pass

def build_pipeline():
    
    pass