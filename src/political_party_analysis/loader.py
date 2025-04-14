from pathlib import Path
from typing import List
from urllib.request import urlretrieve

import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataLoader:
    """Class to load the political parties dataset"""

    data_url: str = "https://www.chesdata.eu/s/CHES2019V3.dta"

    def __init__(self):
        self.party_data = self._download_data()
        self.non_features = []
        self.index = ["party_id", "party", "country"]

    def _download_data(self) -> pd.DataFrame:
        data_path, _ = urlretrieve(
            self.data_url,
            Path(__file__).parents[2].joinpath(*["data", "CHES2019V3.dta"]),
        )
        return pd.read_stata(data_path)

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Write a function to remove duplicates in a dataframe"""
        # Remove duplicates keeping the first occurrence
        return df.drop_duplicates()

    def remove_nonfeature_cols(self, df: pd.DataFrame, non_features: List[str], index: List[str]) -> pd.DataFrame:
        """Write a function to remove certain features cols and set certain cols as indices
        in a dataframe"""
        # Drop non-feature columns and set index
        df_copy = df.copy()
        # Drop columns in non_features list
        if non_features:
            df_copy = df_copy.drop(columns=non_features)
        # Set index columns
        if index:
            df_copy = df_copy.set_index(index)
        return df_copy

    def handle_NaN_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Write a function to handle NaN values in a dataframe"""
        # Drop columns with all NaN values
        df_copy = df.copy()
        df_copy = df_copy.dropna(axis=1, how='all')

        # For remaining columns with some NaN values, fill with column mean
        df_copy = df_copy.apply(lambda x: x.fillna(x.mean()) if x.dtype.kind in 'ifc' else x)

        return df_copy

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Write a function to normalise values in a dataframe. Use StandardScaler."""
        # Use StandardScaler to normalize numerical columns
        df_copy = df.copy()

        # Select only numeric columns
        numeric_cols = df_copy.select_dtypes(include=['int64', 'float64']).columns

        if len(numeric_cols) > 0:
            scaler = StandardScaler()
            df_copy[numeric_cols] = scaler.fit_transform(df_copy[numeric_cols])

        return df_copy

    def preprocess_data(self):
        """Write a function to combine all pre-processing steps for the dataset"""
        # Apply all preprocessing steps sequentially
        # 1. Remove duplicates
        self.party_data = self.remove_duplicates(self.party_data)
        #print("Duplicates removed", self.party_data[:2])

        # 2. Remove non-feature columns and set index
        self.party_data = self.remove_nonfeature_cols(self.party_data, self.non_features, self.index)
        #print("Non-feature columns removed", self.party_data[:2])
        
        # 3. Handle NaN values
        self.party_data = self.handle_NaN_values(self.party_data)
        #print("NaN values handled", self.party_data[:2])
        
        # 4. Scale features
        self.party_data = self.scale_features(self.party_data)
        #print("Features scaled", self.party_data[:2])
        
        return self.party_data
