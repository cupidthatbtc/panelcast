"""Shared fixtures for integration testing.

These fixtures test the integration between major pipeline components:
- Data loading -> cleaning
- Cleaning -> splitting
- Splitting -> feature pipeline
- Model predictions -> evaluation metrics
"""

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from panelcast.data.cleaning import clean_albums
from panelcast.data.ingest import load_raw_albums
from panelcast.data.split import within_entity_temporal_split
from panelcast.features.album_type import AlbumTypeBlock
from panelcast.features.artist import ArtistHistoryBlock
from panelcast.features.base import FeatureContext
from panelcast.features.pipeline import FeaturePipeline
from panelcast.features.temporal import TemporalBlock

# Path to test fixture
FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "raw_all_albums_full.csv"


def generate_synthetic_albums(
    n_artists: int = 10,
    albums_per_artist: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic album dataset for testing.

    Creates a DataFrame with all required columns for feature blocks:
    - Artist, Album, Year, Release_Date, Release_Date_Parsed
    - User_Score, Critic_Score, User_Ratings, Critic_Reviews
    - Album_Type, Genres, date_risk

    Parameters
    ----------
    n_artists : int
        Number of unique artists to generate.
    albums_per_artist : int
        Number of albums per artist.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Synthetic album dataset with required columns.
    """
    rng = np.random.default_rng(seed)

    records = []
    base_year = 2010

    for artist_idx in range(n_artists):
        artist_name = f"Artist_{artist_idx:03d}"
        artist_base_score = rng.uniform(60, 85)

        for album_idx in range(albums_per_artist):
            year = base_year + album_idx
            release_date = pd.Timestamp(
                year=year, month=rng.integers(1, 13), day=rng.integers(1, 28)
            )

            # Scores vary around artist baseline
            user_score = np.clip(artist_base_score + rng.normal(0, 8), 0, 100)
            critic_score = np.clip(artist_base_score + rng.normal(0, 10), 0, 100)

            records.append(
                {
                    "Artist": artist_name,
                    "Album": f"Album_{artist_idx:03d}_{album_idx:02d}",
                    "Year": year,
                    "Release_Date": release_date.strftime("%B %d, %Y"),
                    "Release_Date_Parsed": release_date,
                    "User_Score": user_score,
                    "Critic_Score": critic_score,
                    "User_Ratings": rng.integers(100, 5000),
                    "Critic_Reviews": rng.integers(5, 50),
                    "Album_Type": rng.choice(["Album", "EP", "Mixtape"], p=[0.7, 0.2, 0.1]),
                    "Genres": rng.choice(
                        ["Rock, Indie", "Pop, Dance", "Hip Hop, Rap", "Electronic, Ambient"]
                    ),
                    "date_risk": rng.choice(["low", "medium", "high"], p=[0.8, 0.15, 0.05]),
                    "original_row_id": artist_idx * albums_per_artist + album_idx,
                }
            )

    return pd.DataFrame(records)


def create_mock_posterior_samples(
    n_chains: int = 4,
    n_draws: int = 500,
    param_shapes: dict = None,
    seed: int = 42,
) -> xr.Dataset:
    """Create mock posterior samples as xarray Dataset.

    Parameters
    ----------
    n_chains : int
        Number of MCMC chains.
    n_draws : int
        Number of draws per chain.
    param_shapes : dict
        Mapping of parameter names to their shapes (excluding chain/draw dims).
        Example: {"beta": (5,), "sigma": ()}
    seed : int
        Random seed.

    Returns
    -------
    xr.Dataset
        Mock posterior samples.
    """
    rng = np.random.default_rng(seed)
    param_shapes = param_shapes or {
        "user_beta": (5,),
        "user_sigma_obs": (),
        "user_mu_artist": (),
        "user_sigma_artist": (),
    }

    data_vars = {}
    for name, shape in param_shapes.items():
        full_shape = (n_chains, n_draws) + shape
        data_vars[name] = (
            ("chain", "draw") + tuple(f"dim_{i}" for i in range(len(shape))),
            rng.normal(0, 1, full_shape),
        )

    return xr.Dataset(
        data_vars,
        coords={"chain": range(n_chains), "draw": range(n_draws)},
    )


@pytest.fixture(scope="module")
def cleaned_albums_df() -> pd.DataFrame:
    """Load and clean albums from test fixture.

    Tests ingest -> cleaning integration by:
    1. Loading raw CSV with load_raw_albums()
    2. Applying clean_albums() transformation

    Returns
    -------
    pd.DataFrame
        Cleaned album DataFrame with all derived columns.
    """
    if not FIXTURE_PATH.exists():
        pytest.skip(f"Fixture file not found: {FIXTURE_PATH}")

    raw_df, metadata = load_raw_albums(FIXTURE_PATH, validate=False)
    cleaned = clean_albums(raw_df)
    return cleaned


@pytest.fixture(scope="module")
def synthetic_albums_df() -> pd.DataFrame:
    """Generate synthetic albums for testing.

    Creates a controlled dataset with:
    - 10 artists with 5 albums each (50 total)
    - All required columns for feature blocks
    - Deterministic via fixed seed

    Returns
    -------
    pd.DataFrame
        Synthetic album dataset.
    """
    return generate_synthetic_albums(n_artists=10, albums_per_artist=5, seed=42)


@pytest.fixture(scope="module")
def split_datasets(synthetic_albums_df: pd.DataFrame) -> dict:
    """Split synthetic data into train/val/test.

    Uses within_entity_temporal_split to create temporal splits
    where each artist has albums in all three splits.

    Returns
    -------
    dict
        Dictionary with keys "train", "val", "test" containing DataFrames.
    """
    train, val, test = within_entity_temporal_split(
        synthetic_albums_df,
        entity_col="Artist",
        date_col="Release_Date_Parsed",
        test_albums=1,
        val_albums=1,
        min_train_albums=1,
    )
    return {"train": train, "val": val, "test": test}


@pytest.fixture(scope="module")
def feature_context() -> FeatureContext:
    """Create a feature context for testing.

    Returns
    -------
    FeatureContext
        Shared context with default config and random state.
    """
    return FeatureContext(config={}, random_state=42)


@pytest.fixture(scope="module")
def fitted_feature_pipeline(
    split_datasets: dict,
    feature_context: FeatureContext,
) -> FeaturePipeline:
    """Create and fit a feature pipeline on training data.

    Uses a subset of feature blocks:
    - TemporalBlock: Stateless temporal features
    - ArtistHistoryBlock: LOO-based artist history
    - AlbumTypeBlock: One-hot album type encoding

    Returns
    -------
    FeaturePipeline
        Fitted pipeline ready for transform().
    """
    blocks = [
        TemporalBlock(),
        ArtistHistoryBlock(),
        AlbumTypeBlock(),
    ]
    pipeline = FeaturePipeline(blocks)
    pipeline.fit(split_datasets["train"], feature_context)
    return pipeline


@pytest.fixture
def mock_idata() -> az.InferenceData:
    """Create minimal ArviZ InferenceData for testing.

    Creates an InferenceData with:
    - posterior group with user_beta, user_sigma_obs
    - observed_data group with y
    - sample_stats group with diverging

    Returns
    -------
    az.InferenceData
        Mock inference data for evaluation tests.
    """
    rng = np.random.default_rng(42)
    n_chains = 4
    n_draws = 500
    n_obs = 50

    # Posterior samples
    posterior = xr.Dataset(
        {
            "user_beta": (
                ["chain", "draw", "beta_dim"],
                rng.normal(0, 1, (n_chains, n_draws, 5)),
            ),
            "user_sigma_obs": (
                ["chain", "draw"],
                np.abs(rng.normal(10, 2, (n_chains, n_draws))),
            ),
            "user_mu_artist": (["chain", "draw"], rng.normal(70, 5, (n_chains, n_draws))),
            "user_sigma_artist": (
                ["chain", "draw"],
                np.abs(rng.normal(5, 1, (n_chains, n_draws))),
            ),
        },
        coords={"chain": range(n_chains), "draw": range(n_draws)},
    )

    # Observed data
    observed_data = xr.Dataset(
        {"y": (["y_dim"], rng.uniform(50, 90, n_obs))},
        coords={"y_dim": range(n_obs)},
    )

    # Sample stats (including divergences)
    sample_stats = xr.Dataset(
        {
            "diverging": (["chain", "draw"], np.zeros((n_chains, n_draws), dtype=bool)),
            "tree_depth": (["chain", "draw"], rng.integers(5, 10, (n_chains, n_draws))),
        },
        coords={"chain": range(n_chains), "draw": range(n_draws)},
    )

    return az.InferenceData(
        posterior=posterior,
        observed_data=observed_data,
        sample_stats=sample_stats,
    )


@pytest.fixture
def mock_idata_with_log_lik(mock_idata: az.InferenceData) -> az.InferenceData:
    """Create InferenceData with log_likelihood group for LOO tests.

    Returns
    -------
    az.InferenceData
        Mock inference data with log_likelihood group.
    """
    rng = np.random.default_rng(43)
    n_chains = mock_idata.posterior.sizes["chain"]
    n_draws = mock_idata.posterior.sizes["draw"]
    n_obs = mock_idata.observed_data.sizes["y_dim"]

    # Log-likelihood (negative values, typical for log probs)
    log_likelihood = xr.Dataset(
        {"y": (["chain", "draw", "y_dim"], rng.normal(-5, 1, (n_chains, n_draws, n_obs)))},
        coords={
            "chain": range(n_chains),
            "draw": range(n_draws),
            "y_dim": range(n_obs),
        },
    )

    mock_idata.add_groups(log_likelihood=log_likelihood)
    return mock_idata


@pytest.fixture
def mock_predictions() -> dict:
    """Create mock predictions for evaluation testing.

    Returns
    -------
    dict
        Dictionary with:
        - y_true: True observations (n_obs,)
        - y_pred_samples: Posterior predictive samples (n_samples, n_obs)
        - mean: Posterior mean predictions (n_obs,)
        - hdi_low: Lower bound of 94% HDI (n_obs,)
        - hdi_high: Upper bound of 94% HDI (n_obs,)
    """
    rng = np.random.default_rng(44)
    n_obs = 50
    n_samples = 2000

    # True values centered around 70 with some spread
    y_true = rng.uniform(50, 90, n_obs)

    # Predictions centered on true with some noise
    # Well-calibrated model: predictions should bracket truth ~95% of time
    noise = rng.normal(0, 8, (n_samples, n_obs))
    y_pred_samples = y_true + noise

    mean = y_pred_samples.mean(axis=0)
    hdi_low = np.percentile(y_pred_samples, 3, axis=0)  # 94% HDI lower
    hdi_high = np.percentile(y_pred_samples, 97, axis=0)  # 94% HDI upper

    return {
        "y_true": y_true,
        "y_pred_samples": y_pred_samples,
        "mean": mean,
        "hdi_low": hdi_low,
        "hdi_high": hdi_high,
    }
