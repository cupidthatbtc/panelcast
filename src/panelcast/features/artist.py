"""Artist history feature block (AOTY default of the entity-history block).

Computes artist track record features using expanding windows with shift()
to prevent data leakage. Each album sees only prior albums from that artist.
The generic implementation lives in :mod:`panelcast.features.history`; this
subclass pins the AOTY column names and score specs.
"""

from __future__ import annotations

from .history import (
    DEFAULT_SCORE_SPECS,
    EntityHistoryBlock,
    _compute_trajectory_slope,  # noqa: F401  (re-export for legacy imports)
)


class ArtistHistoryBlock(EntityHistoryBlock):
    """Artist history features using leave-one-out expanding windows.

    Computes prior mean, std, count, and trajectory for each artist,
    excluding the current album (LOO pattern via shift()).

    Required columns: Artist, Release_Date_Parsed, User_Score, Critic_Score, Album

    Output features (9 total):
    - user_prior_mean, user_prior_std, user_prior_count, user_trajectory
    - critic_prior_mean, critic_prior_std, critic_prior_count, critic_trajectory
    - is_debut

    Debut albums have NaN prior statistics which are imputed with global
    means learned from training data during fit().

    Examples
    --------
    >>> block = ArtistHistoryBlock()
    >>> block.fit(train_df, ctx)
    >>> output = block.transform(test_df, ctx)
    >>> output.feature_names
    ['user_prior_mean', 'user_prior_std', ...]
    """

    name = "artist_history"

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(
            params,
            entity_col="Artist",
            date_col="Release_Date_Parsed",
            event_col="Album",
            score_specs=DEFAULT_SCORE_SPECS,
        )


# Backwards compatibility alias
ArtistReputationBlock = ArtistHistoryBlock
