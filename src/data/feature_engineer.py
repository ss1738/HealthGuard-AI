import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple

class FeatureEngineer:
    """Modular class for scaling and feature transformations."""
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def scale_amount(self) -> pd.DataFrame:
        """Scales the 'Amount' feature using StandardScaler."""
        scaler = StandardScaler()
        self.df['Amount_Scaled'] = scaler.fit_transform(self.df['Amount'].values.reshape(-1, 1))
        self.df.drop(['Amount'], axis=1, inplace=True)
        return self.df
        
    def run_feature_pipeline(self) -> pd.DataFrame:
        """Executes the full feature engineering sequence."""
        self.df = self.scale_amount()
        return self.df
        
def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Splits the DataFrame into features (X) and target (y)."""
    TARGET = 'Class'
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    X = X.astype('float64')
    y = y.astype('int')
    return X, y
