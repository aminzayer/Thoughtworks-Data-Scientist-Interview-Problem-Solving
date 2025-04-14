import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture


class DensityEstimator:
    """Class to estimate Density/Distribution of the given data.
    1. Write a function to model the distribution of the political party dataset
    2. Write a function to randomly sample 10 parties from this distribution
    3. Map the randomly sampled 10 parties back to the original higher dimensional
    space as per the previously used dimensionality reduction technique.
    """

    def __init__(self, data: pd.DataFrame, dim_reducer, high_dim_feature_names):
        self.data = data
        self.dim_reducer_model = dim_reducer.model
        self.feature_names = high_dim_feature_names
        self.model = None
        self.labels = None
        self.means = None
        self.covariances = None

    def fit(self, n_components=4):
        """Fit a Gaussian Mixture Model to the data"""
        self.model = GaussianMixture(n_components=n_components)
        self.model.fit(self.data)
        self.labels = self.model.predict(self.data)
        self.means = self.model.means_
        self.covariances = self.model.covariances_
        return self.labels, self.means, self.covariances

    def sample(self, n_samples=10):
        """Sample from the fitted distribution"""
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        # Generate random samples from the fitted GMM
        samples, _ = self.model.sample(n_samples=n_samples)

        # Convert to DataFrame
        sampled_df = pd.DataFrame(samples, columns=[f'component_{i+1}' for i in range(self.data.shape[1])])

        return sampled_df

    def inverse_transform(self, low_dim_data):
        """Map the low-dimensional data back to high-dimensional space"""
        if self.dim_reducer_model is None:
            raise ValueError("Dimensionality reducer model not set.")

        # Use the inverse_transform method of PCA to map back to original space
        high_dim_data = self.dim_reducer_model.inverse_transform(low_dim_data)

        # Convert to DataFrame with original feature names
        result = pd.DataFrame(high_dim_data, columns=self.feature_names)

        return result
