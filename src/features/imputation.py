from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import numpy as np

class IndicatorImputer(TransformerMixin, BaseEstimator):
    def fit(self, X: pd.DataFrame, y=None):
        # Assuming None is only at the start of the series,
        # by counting None, we get the index of the first non-None element.
        # We get an array of indices, by which we locate corresponding rows
        # in the dataset. The diagonal of these 'matrix' are the first non-None
        # values
        self._values_to_impute = dict(zip(X.columns,
                                          np.diag(X.iloc[X.isnull().sum()])))

    def transform(self, X: pd.DataFrame):
        return X.fillna(self._values_to_impute)
