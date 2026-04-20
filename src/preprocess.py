from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class DaysEmployedFixer(BaseEstimator, TransformerMixin):
    '''Replace anomaly value 365243 (representing unemployment/retirement) with nan'''
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        df = X.copy()
        df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
        
        return df

        



class DaysColumnsTransformer(BaseEstimator, TransformerMixin):

    pass

class FlagEngineer(BaseEstimator, TransformerMixin):
    pass

class RatioEngineer(BaseEstimator, TransformerMixin):
    pass

class CategoryGrouper(BaseEstimator, TransformerMixin):
    pass

def build_pipeline():
    
    pass