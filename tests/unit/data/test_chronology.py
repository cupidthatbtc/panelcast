import pandas as pd
import pytest

from panelcast.config.descriptor import DatasetDescriptor
from panelcast.data.alignment import ROW_ID_COL
from panelcast.data.chronology import DATE_MISSING_COL, normalize_chronology
from panelcast.data.split import within_entity_temporal_split
from panelcast.features.base import FeatureContext
from panelcast.features.history import EntityHistoryBlock
from panelcast.features.temporal import TemporalBlock
from panelcast.pipelines.evaluate import _normalize_optional_frame
from panelcast.pipelines.predict_next import _normalize_training_chronology
from panelcast.pipelines.train_bayes import prepare_model_data


def _mixed_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity": ["b", "a", "a", "a", "a"],
            "event": ["z", "b", "a", "missing", "a"],
            "date": ["2018", "2020-01-01", "2020-01-01T00:00:00Z", None, "2019-06"],
            "value": [0, 2, 1, 3, 4],
        },
        index=[50, 40, 30, 20, 10],
    )


def test_mixed_precision_missing_and_ties_have_one_stable_order():
    frame = _mixed_rows()
    expected = [20, 50, 10, 30, 40]
    assert normalize_chronology(
        frame, entity_col="entity", date_col="date", event_col="event"
    ).index.tolist() == expected
    assert normalize_chronology(
        frame.sample(frac=1, random_state=9),
        entity_col="entity",
        date_col="date",
        event_col="event",
    ).index.tolist() == expected


def test_missing_dates_are_signalled_or_rejected():
    normalized = normalize_chronology(
        _mixed_rows(), entity_col="entity", date_col="date", event_col="event"
    )
    assert normalized.loc[20, DATE_MISSING_COL] == 1
    with pytest.raises(ValueError, match="missing or invalid"):
        normalize_chronology(
            _mixed_rows(),
            entity_col="entity",
            date_col="date",
            event_col="event",
            reject_missing=True,
        )


def test_tie_order_depends_only_on_immutable_row_identity():
    frame = pd.DataFrame(
        {
            "entity": ["a"] * 4,
            "event": ["same"] * 4,
            "date": ["2020-01-01"] * 4,
            "target": [10.0, 20.0, 30.0, 40.0],
            "feature": pd.Series([1, 2, 3, 4], dtype="int64"),
            ROW_ID_COL: [30, 10, 40, 20],
        }
    )

    def ordered_ids(candidate: pd.DataFrame) -> list[int]:
        return normalize_chronology(
            candidate, entity_col="entity", date_col="date", event_col="event"
        )[ROW_ID_COL].tolist()

    expected = [10, 20, 30, 40]
    assert ordered_ids(frame) == expected
    masked = frame.assign(target=float("nan"))
    shuffled = frame.sample(frac=1, random_state=7).reset_index(drop=True)
    round_tripped = frame.assign(feature=frame["feature"].astype("float64"))
    assert ordered_ids(masked) == expected
    assert ordered_ids(shuffled) == expected
    assert ordered_ids(round_tripped) == expected


def test_consumers_share_the_canonical_event_order():
    frame = pd.DataFrame(
        {
            "entity": ["a"] * 5,
            "event": ["z", "missing", "early", "b", "a"],
            "date": ["2020", None, "2019", "2020-01-01", "2020-01-01"],
            "year": [2020, 0, 2019, 2020, 2020],
            "date_risk": ["low", "high", "low", "low", "low"],
            "target": [50.0, 10.0, 20.0, 40.0, 30.0],
            "n_reviews": [10] * 5,
            "feature": [5.0, 1.0, 2.0, 4.0, 3.0],
            ROW_ID_COL: [50, 10, 30, 20, 40],
        }
    )
    expected = [10, 30, 40, 20, 50]
    normalized = normalize_chronology(
        frame, entity_col="entity", date_col="date", event_col="event"
    )
    assert normalized[ROW_ID_COL].tolist() == expected

    ctx = FeatureContext(config={}, random_state=42)
    temporal = TemporalBlock(
        entity_col="entity", date_col="date", year_col="year", event_col="event"
    ).fit(frame, ctx)
    temporal_output = temporal.transform(frame, ctx).data
    assert temporal_output.loc[normalized.index, "album_sequence"].tolist() == [1, 2, 3, 4, 5]

    history = EntityHistoryBlock(
        entity_col="entity",
        date_col="date",
        event_col="event",
        score_specs=[("target", "score")],
    ).fit(frame, ctx)
    history_output = history.transform(frame, ctx).data
    assert history_output.loc[normalized.index, "score_prior_count"].tolist() == [0, 1, 2, 3, 4]

    descriptor = DatasetDescriptor(
        entity_col="entity",
        event_col="event",
        date_col="date",
        parsed_date_col="date",
        year_col="year",
        target_col="target",
        n_obs_col="n_reviews",
        secondary_target_col=None,
        secondary_prefix=None,
        secondary_n_obs_col=None,
    )
    model_args, _ = prepare_model_data(
        frame, ["feature"], min_albums_filter=1, descriptor=descriptor
    )
    assert model_args["album_seq"].tolist() == [1, 2, 3, 4, 5]
    assert model_args["y"].tolist() == normalized["target"].tolist()

    train, validation, test = within_entity_temporal_split(
        frame,
        entity_col="entity",
        date_col="date",
        event_col="event",
        test_albums=1,
        val_albums=1,
        min_train_albums=2,
    )
    assert train[ROW_ID_COL].tolist() == expected[:3]
    assert validation[ROW_ID_COL].tolist() == [20]
    assert test[ROW_ID_COL].tolist() == [50]

    ds = {
        "entity_col": "entity",
        "event_col": "event",
        "parsed_date_col": "date",
        "date_col": "date",
    }
    evaluated = _normalize_optional_frame(frame, ds, "date")
    predicted = _normalize_training_chronology(frame, "entity", ds)
    assert evaluated is not None
    assert evaluated[ROW_ID_COL].tolist() == expected
    assert predicted[ROW_ID_COL].tolist() == expected


def test_invalid_entity_or_row_identity_is_rejected():
    frame = pd.DataFrame(
        {
            "entity": ["a", None],
            "event": ["x", "y"],
            "date": ["2020-01-01", "2020-01-02"],
            ROW_ID_COL: [1, 2],
        }
    )
    with pytest.raises(ValueError, match="non-missing 'entity'"):
        normalize_chronology(frame, entity_col="entity", date_col="date", event_col="event")

    frame["entity"] = ["a", "b"]
    for invalid_ids in (["one", "two"], [1.5, 2.0], [1, float("inf")]):
        frame[ROW_ID_COL] = invalid_ids
        with pytest.raises(ValueError, match="integer 'original_row_id'"):
            normalize_chronology(
                frame, entity_col="entity", date_col="date", event_col="event"
            )


def test_unidentified_exact_tie_is_rejected():
    frame = pd.DataFrame(
        {"entity": ["a", "a"], "date": ["2020", "2020"], "event": ["x", "x"]}
    )
    with pytest.raises(ValueError, match=ROW_ID_COL):
        normalize_chronology(frame, entity_col="entity", date_col="date", event_col="event")
