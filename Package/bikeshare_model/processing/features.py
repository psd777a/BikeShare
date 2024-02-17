import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class year_month(BaseEstimator, TransformerMixin):
    """
    Extracts Year and Month form Date
    """

    def __init__(self, variable: str):
        # YOUR CODE HERE
        self.variable = variable

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # YOUR CODE HERE
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # YOUR CODE HERE
        X = X.copy()
        X[self.variable] = pd.to_datetime(X[self.variable], format="%Y-%m-%d")
        X["Year"] = X[self.variable].dt.year
        X["Month"] = X[self.variable].dt.month_name()
        return X


class WeekdayImputer(BaseEstimator, TransformerMixin):
    """Impute missing values in 'weekday' column by extracting dayname from 'dteday' column"""

    def __init__(self, variable: str, date_var: str):
        # YOUR CODE HERE
        self.variable = variable
        self.date_var = date_var

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # YOUR CODE HERE
        X = X.copy()
        null_row = X[X[self.variable].isnull() == True].index
        X.loc[null_row, self.variable] = (
            X.loc[null_row, self.date_var].dt.day_name().apply(lambda x: x[:3])
        )
        X = X.drop([self.date_var], axis=1)

        return X


class WeathersitImputer(BaseEstimator, TransformerMixin):
    """Impute missing values in 'weathersit' column by replacing them with the most frequent category value"""

    def __init__(self, variable: str):
        # YOUR CODE HERE
        self.variable = variable
        self.fill : str

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # YOUR CODE HERE
        self.fill = X[self.variable].mode()[0]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # YOUR CODE HERE
        X = X.copy()
        X[self.variable] = X[self.variable].fillna(self.fill)
        return X


class Mapper(BaseEstimator, TransformerMixin):
    """
    Ordinal categorical variable mapper:
    Treat column as Ordinal categorical variable, and assign values accordingly
    """

    def __init__(self, variable: str, mappings: dict):
        # YOUR CODE HERE
        self.variable = variable
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # YOUR CODE HERE
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # YOUR CODE HERE
        X = X.copy()
        X[self.variable] = X[self.variable].map(self.mappings).astype(int)
        return X


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values:
        - to upper-bound, if the value is higher than upper-bound, or
        - to lower-bound, if the value is lower than lower-bound respectively.
    """

    def __init__(self, variable: str):
        # YOUR CODE HERE
        if not isinstance(variable, str):
            raise ValueError("variables should be a str")
        self.variable = variable
        self.lb : float
        self.ub : float

    def fit(self, X: pd.DataFrame, y=None):
        # YOUR CODE HERE
        q1 = np.percentile(X[self.variable], 25)
        q3 = np.percentile(X[self.variable], 75)

        iqr = q3 - q1

        self.lb = q1 - 1.5 * iqr
        self.ub = q3 + 1.5 * iqr
        return self

    def transform(self, X) -> pd.DataFrame:
        # YOUR CODE HERE
        X = X.copy()  # so that we do not over-write the original dataframe
        X.loc[X[self.variable] < self.lb, self.variable] = self.lb
        X.loc[X[self.variable] > self.ub, self.variable] = self.ub
        return X


class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    """One-hot encode weekday column"""

    def __init__(self, variable: str):
        # YOUR CODE HERE
        if not isinstance(variable, str):
            raise ValueError("variables should be a str")
        self.variable = variable

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # YOUR CODE HERE
        self.ohe = OneHotEncoder(dtype=int, drop="first")
        self.ohe.fit(X[self.variable].values.reshape(-1, 1))
        self.col = [
            self.variable + "_" + x[3:] for x in self.ohe.get_feature_names_out()
        ]
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        # YOUR CODE HERE
        X = X.copy()
        out = pd.DataFrame(
            self.ohe.transform(X[self.variable].values.reshape(-1, 1)).toarray(),
            columns=self.col,
        )
        X = pd.concat([X, out], axis=1)
        X = X.drop([self.variable], axis=1)
        return X
