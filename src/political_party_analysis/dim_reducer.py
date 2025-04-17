import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


class DimensionalityReducer:
    """Class to model a dimensionality reduction method for the given dataset.
    1. Write a function to convert the high dimensional data to 2 dimensional.
    """

    def __init__(self, data: pd.DataFrame, n_components: int = 2):
        self.n_components = n_components
        self.data = data
        self.model = PCA(n_components=self.n_components)

    def transform(self) -> pd.DataFrame:
        """Transform numeric data to lower dimensions using PCA."""
        numeric_data = self.data.select_dtypes(include=['int32', 'int64', 'float32', 'float64'])

        if numeric_data.empty:
            raise ValueError("No numeric columns found for PCA. Got columns: " + str(self.data.dtypes))

        transformed_data = self.model.fit_transform(numeric_data)

        result = pd.DataFrame(transformed_data, index=numeric_data.index, columns=[f'component_{i+1}' for i in range(self.n_components)])
        
        return result
