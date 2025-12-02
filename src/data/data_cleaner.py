import pandas as pd

class DataCleaner:
    """Modular class for data cleaning (duplicates, missing values, dropping 'Time')."""
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def handle_missing_values(self) -> pd.DataFrame:
        """Addresses missing values by dropping negligible rows."""
        self.df.dropna(inplace=True)
        return self.df

    def drop_duplicates(self) -> pd.DataFrame:
        """Removes any fully duplicated rows."""
        self.df.drop_duplicates(inplace=True)
        return self.df

    def standardize_features(self) -> pd.DataFrame:
        """Drops non-essential columns like 'Time'."""
        self.df.drop('Time', axis=1, inplace=True, errors='ignore')
        return self.df

    def run_cleaning_pipeline(self) -> pd.DataFrame:
        """Executes the full cleaning sequence."""
        self.df = self.drop_duplicates()
        self.df = self.handle_missing_values()
        self.df = self.standardize_features()
        return self.df
