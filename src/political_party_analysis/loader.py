from pathlib import Path
from typing import List
from urllib.request import urlretrieve

import pandas as pd
from sklearn.impute import KNNImputer
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
        """
        Remove duplicate rows from the DataFrame, keeping the first occurrence.
        """
        # Drop duplicate rows while retaining the first entry
        return df.drop_duplicates()
        # ➤ Reason: Ensures no repeated observations distort statistics or model learning.

    def remove_nonfeature_cols(
        self, df: pd.DataFrame, non_features: List[str], index: List[str]
    ) -> pd.DataFrame:
        """
        Remove specified non-feature columns and set given columns as DataFrame index.
        """
        df_copy = df.copy()

        # Drop user-specified non-feature columns (e.g., metadata or IDs)
        if non_features:
            df_copy = df_copy.drop(columns=non_features)

        # Set designated index columns (e.g., 'party', 'country') for clarity and future joins
        if index:
            df_copy = df_copy.set_index(index)

        return df_copy
        # ➤ Reason: Keeps only relevant features and sets a logical index for tracking/merging.

    def handle_NaN_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean missing values in the DataFrame using a specified strategy.

        Steps:
        1. Drop columns with excessive missing values.
        2. Fill remaining missing values using the selected imputation strategy.

        The strategy used can be: mean, median, mode, forward fill, backward fill, constant, or KNN
        imputation.
        """

        # Define the imputation strategy to use (can be changed as needed)
        strategy = "KNN"

        # Step 1: Drop columns with more than 50% missing values
        # Reason: Columns with too many missing values are often not informative and may introduce
        # noise
        threshold = int(len(df) / 2)  # Allow up to 50% missing data
        df_copy = df.copy()  # Work on a copy to avoid modifying the original data
        df_copy = df_copy.dropna(axis=1, thresh=threshold)

        # Step 2: Apply the chosen imputation strategy to remaining missing values
        # Reason: Each method handles missingness differently, suitable for different data types or
        # distributions

        if strategy == "mean":
            # Fill missing numeric values with the column-wise mean
            # Reason: Assumes missing values are missing at random and that the mean represents
            # central tendency
            df_copy = df_copy.apply(lambda x: x.fillna(x.mean()) if x.dtype.kind in "ifc" else x)

        elif strategy == "median":
            # Fill missing numeric values with the column-wise median
            # Reason: More robust to outliers than the mean
            df_copy = df_copy.apply(lambda x: x.fillna(x.median()) if x.dtype.kind in "ifc" else x)

        elif strategy == "mode":
            # Fill missing values with the most frequent value (mode) for each column
            # Reason: Best for categorical or low-cardinality features
            df_copy = df_copy.apply(lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x)

        elif strategy == "ffill":
            # Forward fill: propagate last valid value forward
            # Reason: Useful for time-series or sequential data
            df_copy = df_copy.fillna(method="ffill")

        elif strategy == "bfill":
            # Backward fill: propagate next valid value backward
            # Reason: Similar to forward fill but fills in the opposite direction
            df_copy = df_copy.fillna(method="bfill")

        elif strategy == "constant":
            # Fill all missing values with a constant (here, zero)
            # Reason: Ensures completeness but may distort distributions if not handled carefully
            df_copy = df_copy.fillna(value=0)

        elif strategy == "KNN":
            # Use K-Nearest Neighbors imputation based on feature similarity
            # Reason: More sophisticated approach; captures complex patterns in the data
            imputer = KNNImputer(n_neighbors=3, weights="distance")
            df_copy = pd.DataFrame(
                imputer.fit_transform(df_copy), columns=df_copy.columns, index=df_copy.index
            )

        # Return the cleaned DataFrame
        return df_copy
        # ➤ Reason: Preserves useful columns while handling missing data in a statistically sound
        # way.

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize numerical columns using StandardScaler (zero mean, unit variance).
        """
        df_copy = df.copy()

        # Identify numerical columns
        numeric_cols = df_copy.select_dtypes(
            include=["int32", "int64", "float32", "float64"]
        ).columns

        scaler = StandardScaler()
        df_copy[numeric_cols] = scaler.fit_transform(df_copy[numeric_cols])

        return df_copy
        # ➤ Reason: Normalization improves model convergence and comparability across features.

    def preprocess_data(self):
        """
        Apply all preprocessing steps in order:
        1. Remove duplicates
        2. Remove non-feature columns, set index
        3. Handle missing values
        4. Normalize numeric features
        """
        # Step 1: Eliminate duplicate rows
        self.party_data = self.remove_duplicates(self.party_data)

        # Step 2: Remove specified non-feature columns and define DataFrame index
        self.party_data = self.remove_nonfeature_cols(
            self.party_data, self.non_features, self.index
        )

        # Step 3: Clean NaN values appropriately (drop or impute)
        self.party_data = self.handle_NaN_values(self.party_data)

        # Step 4: Normalize numeric columns for modeling
        self.party_data = self.scale_features(self.party_data)

        return self.party_data
        # ➤ Reason: Ensures clean, consistent, and normalized data ready for ML pipelines.
