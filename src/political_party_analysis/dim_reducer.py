import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


class DimensionalityReducer:
    """Class to model a dimensionality reduction method for the given dataset.
    1. Write a function to convert the high dimensional data to 2 dimensional.
    """

    def __init__(self, data: pd.DataFrame, n_components: int = 2):
        # Store number of target components and input dataset
        self.n_components = n_components
        self.data = data

        # Initialize PCA model with specified dimensionality
        self.model = PCA(n_components=self.n_components)
        # ➤ Reason: Allows flexible projection into lower-dimensional space for visualization or clustering

    def transform(self) -> pd.DataFrame:
        """Transform numeric data to lower dimensions using PCA."""

        # # Select only numeric columns from the DataFrame
        # numeric_data = self.data.select_dtypes(include=['int32', 'int64', 'float32', 'float64'])

        # # Raise an error if no numeric features are found
        # if numeric_data.empty:
        #     raise ValueError("No numeric columns found for PCA. Got columns: " + str(self.data.dtypes))
        # # ➤ Reason: PCA requires continuous numeric input; other types would cause failure

        # Apply PCA transformation to reduce dimensionality
        transformed_data = self.model.fit_transform(self.data)
        # ➤ Reason: Captures the maximum variance in the dataset with fewer components

        # Return the reduced data as a new DataFrame with meaningful column names
        result = pd.DataFrame(transformed_data, index=self.data.index, columns=[f'component_{i+1}' for i in range(self.n_components)])
        # ➤ Reason: Preserves index alignment and adds readable column labels for downstream use

        return result
