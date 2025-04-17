import pandas as pd

from sklearn.mixture import GaussianMixture


class DensityEstimator:
    """Class to estimate Density/Distribution of the given data.
    1. Write a function to model the distribution of the political party dataset
    2. Write a function to randomly sample 10 parties from this distribution
    3. Map the randomly sampled 10 parties back to the original higher dimensional
    space as per the previously used dimensionality reduction technique.
    """

    def __init__(self, data: pd.DataFrame, dim_reducer, high_dim_feature_names):
        # Store the low-dimensional input data (e.g., PCA output)
        self.data = data

        # Save the PCA model from the dimensionality reducer (needed for inverse mapping)
        self.dim_reducer_model = dim_reducer.model

        # Save original high-dimensional feature names for reconstruction
        self.feature_names = high_dim_feature_names

        # Initialize placeholders for the GMM model and its outputs
        self.model = None
        self.labels = None
        self.means = None
        self.covariances = None
        # ➤ Reason: Enables deferred fitting and storing cluster info after model training

    def fit(self, n_components=4):
        """Fit a Gaussian Mixture Model (GMM) to the low-dimensional data."""

        # Initialize and fit the GMM to the PCA-reduced data
        self.model = GaussianMixture(n_components=n_components)
        self.model.fit(self.data)

        # Predict cluster labels for each sample
        self.labels = self.model.predict(self.data)

        # Extract cluster centers (means) and covariances
        self.means = self.model.means_
        self.covariances = self.model.covariances_

        return self.labels, self.means, self.covariances
        # ➤ Reason: Enables soft clustering and density estimation in latent space

    def sample(self, n_samples=10):
        """Sample synthetic points from the learned GMM distribution."""

        # Ensure that the model is fitted before sampling
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        # Generate n_samples synthetic points from the GMM
        samples, _ = self.model.sample(n_samples=n_samples)

        # Wrap results in a DataFrame with synthetic component labels
        sampled_df = pd.DataFrame(
            samples, columns=[f"component_{i + 1}" for i in range(self.data.shape[1])]
        )

        return sampled_df
        # ➤ Reason: Allows simulation, data augmentation, or visualization of learned density

    def inverse_transform(self, low_dim_data):
        """Map low-dimensional latent points back to high-dimensional input space."""
        # Check that the dimensionality reduction model (e.g., PCA) is available
        if self.dim_reducer_model is None:
            raise ValueError("Dimensionality reducer model not set.")
        # ➤ Reason: Inverse transform requires access to the original PCA model

        # Use PCA’s inverse_transform to project points back to the original feature space
        high_dim_data = self.dim_reducer_model.inverse_transform(low_dim_data)

        # Convert the result into a DataFrame using the original feature names
        result = pd.DataFrame(high_dim_data, columns=self.feature_names)

        return result
        # ➤ Reason: Enables interpretation of clusters or samples in the context of original
        # features
