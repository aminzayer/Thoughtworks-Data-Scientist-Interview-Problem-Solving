import pandas as pd
import pytest
from political_party_analysis.dim_reducer import DimensionalityReducer


@pytest.fixture
def mock_df() -> pd.DataFrame:
    df = pd.DataFrame(
        data={
            "col1": [-1.225, 0, 1.225],
            "col2": [-1.175, -0.1, 1.257],
            "col3": [-1.019, -0.340, 1.359],
        },
        index=[0, 1, 2],
    )
    df.index.name = "id"
    return df


def test_dimensionality_reducer(mock_df: pd.DataFrame):
    """Test dimensionality reduction using PCA"""
    # error for it
    # dim_reducer = DimensionalityReducer("PCA", mock_df, n_components=2)
    # change to this
    dim_reducer = DimensionalityReducer(mock_df, n_components=2)
    transformed_data = dim_reducer.transform()
    assert transformed_data.shape == (mock_df.shape[0], 2)
