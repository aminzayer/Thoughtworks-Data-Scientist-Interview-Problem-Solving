from pathlib import Path
from matplotlib import pyplot

from political_party_analysis.loader import DataLoader
from political_party_analysis.visualization import (
    scatter_plot,
    plot_density_estimation_results,
    plot_finnish_parties,
)
from political_party_analysis.dim_reducer import DimensionalityReducer
from political_party_analysis.estimator import DensityEstimator

if __name__ == "__main__":

    # === 1. Data Preprocessing ===
    # Load and clean the input dataset using a custom DataLoader class.
    # This step ensures missing values, invalid entries, and unwanted noise are handled before \
    # analysis.
    data_loader = DataLoader()
    processed_data = data_loader.preprocess_data()
    print(f"Processed data shape: {processed_data.shape}")
    # ➤ Reason: Ensures high-quality, structured input for downstream ML steps.

    # === 2. Dimensionality Reduction (PCA) ===
    # Reduce the number of features using PCA (Principal Component Analysis).
    # Converts high-dimensional data into 2D for visualization while preserving most variance.
    dim_reducer = DimensionalityReducer(data=processed_data)
    reduced_dim_data = dim_reducer.transform()
    print(f"Reduced dimension data shape: {reduced_dim_data.shape}")
    # ➤ Reason: Simplifies data for visual interpretation and improves clustering performance.

    # === 3. Scatter Plot of Reduced Data ===
    # Optional visualization of PCA-reduced data to verify structure and separability.
    pyplot.figure()
    splot = pyplot.subplot()
    scatter_plot(
        reduced_dim_data,
        color="r",
        splot=splot,
        label="dim reduced data",
    )
    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "dim_reduced_data.png"]))
    # ➤ Reason: Helps verify how well PCA captured the variance in a 2D space.

    # === 4. Density Estimation (e.g. Gaussian Mixture Model) ===
    # Fit a density model to the reduced-dimensional space to identify clusters or latent \
    # structures.
    # Also passes the original feature names to enable back-transformation later.
    feature_names = processed_data.columns
    print(feature_names)
    density_estimator = DensityEstimator(
        data=reduced_dim_data, dim_reducer=dim_reducer, high_dim_feature_names=feature_names
    )

    # Fit the GMM and extract key outputs
    labels, means, covariances = density_estimator.fit(n_components=4)
    # ➤ Reason: Models the distribution of political parties in latent space with probabilistic
    # clusters.

    # === 5. Plot Density Estimation Results ===
    # Visualize the clustering results over PCA-reduced coordinates.
    plot_density_estimation_results(
        reduced_dim_data, labels, means, covariances, "Density Estimation of Political Parties"
    )
    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "density_estimation.png"]))
    # ➤ Reason: Allows visual validation of how clusters align in latent space.

    # === 6. Plot Left vs Right Wing Parties ===
    pyplot.figure()
    splot = pyplot.subplot()

    # Merge PCA results with original ideological axis (lrgen) to classify party orientation.
    # Convention: values < 5 → Left-wing, values > 5 → Right-wing
    combined = reduced_dim_data.copy()
    combined["lrgen"] = processed_data["lrgen"]
    print(combined[:10])
    # Create masks for left and right wing segments
    left_wing_data = combined[combined["lrgen"] < 0].drop(columns=["lrgen"])
    right_wing_data = combined[combined["lrgen"] > 0].drop(columns=["lrgen"])

    print(processed_data["lrgen"].describe())
    print(processed_data["lrgen"].head())
    print(f"Left wing: {len(left_wing_data)}, Right wing: {len(right_wing_data)}")

    # Visualize left-wing parties in red
    scatter_plot(left_wing_data, color="r", splot=splot, label="Left Wing")

    # Visualize right-wing parties in blue
    scatter_plot(right_wing_data, color="b", splot=splot, label="Right Wing")

    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "left_right_parties.png"]))
    pyplot.title("Lefty/righty parties")
    # ➤ Reason: Compares political alignment visually across 2D latent space.

    # === 7. Plot Finnish Parties ===
    # Extract and plot Finnish parties only for regional focus.
    pyplot.figure()
    splot = pyplot.subplot()
    plot_finnish_parties(reduced_dim_data, splot=splot)
    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "finnish_parties.png"]))
    # ➤ Reason: Enables country-specific insights, useful for comparative political analysis.

    # === End of Analysis ===
    print("Analysis Complete")
