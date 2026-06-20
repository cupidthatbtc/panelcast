"""Tests for feature error types and frozen state dataclasses.

Covers NotFittedError, FittedVocabulary, and FittedStatistics
including edge cases and boundary conditions.
"""

from dataclasses import FrozenInstanceError

import pytest

from panelcast.features.errors import (
    FittedStatistics,
    FittedVocabulary,
    NotFittedError,
)

# ---------------------------------------------------------------------------
# NotFittedError tests
# ---------------------------------------------------------------------------


class TestNotFittedError:
    def test_is_value_error(self):
        with pytest.raises(ValueError):
            raise NotFittedError("test")

    def test_is_attribute_error(self):
        with pytest.raises(AttributeError):
            raise NotFittedError("test")

    def test_is_not_fitted_error(self):
        with pytest.raises(NotFittedError):
            raise NotFittedError("test")

    def test_preserves_message(self):
        err = NotFittedError("custom message")
        assert str(err) == "custom message"

    def test_empty_message(self):
        err = NotFittedError("")
        assert str(err) == ""

    def test_no_args(self):
        err = NotFittedError()
        assert isinstance(err, NotFittedError)

    def test_isinstance_checks(self):
        err = NotFittedError("test")
        assert isinstance(err, ValueError)
        assert isinstance(err, AttributeError)
        assert isinstance(err, NotFittedError)

    def test_catch_as_exception(self):
        with pytest.raises(Exception):
            raise NotFittedError("test")


# ---------------------------------------------------------------------------
# FittedVocabulary tests
# ---------------------------------------------------------------------------


class TestFittedVocabulary:
    def test_basic_creation(self):
        vocab = FittedVocabulary(
            categories=("A", "B", "C"),
            category_to_idx={"A": 0, "B": 1, "C": 2},
            unknown_idx=-1,
        )
        assert vocab.categories == ("A", "B", "C")
        assert vocab.category_to_idx == {"A": 0, "B": 1, "C": 2}
        assert vocab.unknown_idx == -1

    def test_is_frozen(self):
        vocab = FittedVocabulary(
            categories=("A",),
            category_to_idx={"A": 0},
        )
        with pytest.raises(FrozenInstanceError):
            vocab.categories = ("X",)

    def test_frozen_unknown_idx(self):
        vocab = FittedVocabulary(
            categories=("A",),
            category_to_idx={"A": 0},
        )
        with pytest.raises(FrozenInstanceError):
            vocab.unknown_idx = 99

    def test_default_unknown_idx(self):
        vocab = FittedVocabulary(
            categories=("A",),
            category_to_idx={"A": 0},
        )
        assert vocab.unknown_idx == -1

    def test_default_category_to_idx(self):
        vocab = FittedVocabulary(categories=("A", "B"))
        assert vocab.category_to_idx == {}

    def test_custom_unknown_idx(self):
        vocab = FittedVocabulary(
            categories=("A",),
            category_to_idx={"A": 0},
            unknown_idx=99,
        )
        assert vocab.unknown_idx == 99

    def test_encode_all_known(self):
        vocab = FittedVocabulary(
            categories=("A", "B", "C"),
            category_to_idx={"A": 0, "B": 1, "C": 2},
        )
        result = vocab.encode(["A", "B", "C"])
        assert result == [0, 1, 2]

    def test_encode_with_unknown(self):
        vocab = FittedVocabulary(
            categories=("A", "B"),
            category_to_idx={"A": 0, "B": 1},
            unknown_idx=-1,
        )
        result = vocab.encode(["A", "X", "B", "Y"])
        assert result == [0, -1, 1, -1]

    def test_encode_all_unknown(self):
        vocab = FittedVocabulary(
            categories=("A",),
            category_to_idx={"A": 0},
            unknown_idx=-1,
        )
        result = vocab.encode(["X", "Y", "Z"])
        assert result == [-1, -1, -1]

    def test_encode_empty_list(self):
        vocab = FittedVocabulary(
            categories=("A",),
            category_to_idx={"A": 0},
        )
        result = vocab.encode([])
        assert result == []

    def test_encode_single_element(self):
        vocab = FittedVocabulary(
            categories=("A", "B"),
            category_to_idx={"A": 0, "B": 1},
        )
        result = vocab.encode(["B"])
        assert result == [1]

    def test_encode_repeated_values(self):
        vocab = FittedVocabulary(
            categories=("A", "B"),
            category_to_idx={"A": 0, "B": 1},
        )
        result = vocab.encode(["A", "A", "B", "A"])
        assert result == [0, 0, 1, 0]

    def test_encode_preserves_custom_unknown_idx(self):
        vocab = FittedVocabulary(
            categories=("A",),
            category_to_idx={"A": 0},
            unknown_idx=42,
        )
        result = vocab.encode(["A", "UNKNOWN"])
        assert result == [0, 42]

    def test_empty_vocabulary(self):
        vocab = FittedVocabulary(categories=(), category_to_idx={})
        result = vocab.encode(["anything"])
        assert result == [-1]

    def test_equality(self):
        vocab1 = FittedVocabulary(
            categories=("A", "B"),
            category_to_idx={"A": 0, "B": 1},
            unknown_idx=-1,
        )
        vocab2 = FittedVocabulary(
            categories=("A", "B"),
            category_to_idx={"A": 0, "B": 1},
            unknown_idx=-1,
        )
        assert vocab1 == vocab2


# ---------------------------------------------------------------------------
# FittedStatistics tests
# ---------------------------------------------------------------------------


class TestFittedStatistics:
    def test_basic_creation(self):
        stats = FittedStatistics(mean=50.0, std=10.0, min_val=20.0, max_val=80.0)
        assert stats.mean == 50.0
        assert stats.std == 10.0
        assert stats.min_val == 20.0
        assert stats.max_val == 80.0

    def test_is_frozen(self):
        stats = FittedStatistics(mean=50.0, std=10.0, min_val=20.0, max_val=80.0)
        with pytest.raises(FrozenInstanceError):
            stats.mean = 100.0

    def test_frozen_std(self):
        stats = FittedStatistics(mean=50.0, std=10.0, min_val=20.0, max_val=80.0)
        with pytest.raises(FrozenInstanceError):
            stats.std = 99.0

    def test_frozen_min_val(self):
        stats = FittedStatistics(mean=50.0, std=10.0, min_val=20.0, max_val=80.0)
        with pytest.raises(FrozenInstanceError):
            stats.min_val = 0.0

    def test_frozen_max_val(self):
        stats = FittedStatistics(mean=50.0, std=10.0, min_val=20.0, max_val=80.0)
        with pytest.raises(FrozenInstanceError):
            stats.max_val = 200.0

    def test_zero_values(self):
        stats = FittedStatistics(mean=0.0, std=0.0, min_val=0.0, max_val=0.0)
        assert stats.mean == 0.0
        assert stats.std == 0.0

    def test_negative_values(self):
        stats = FittedStatistics(mean=-5.0, std=2.0, min_val=-10.0, max_val=-1.0)
        assert stats.mean == -5.0
        assert stats.min_val == -10.0

    def test_equality(self):
        stats1 = FittedStatistics(mean=50.0, std=10.0, min_val=20.0, max_val=80.0)
        stats2 = FittedStatistics(mean=50.0, std=10.0, min_val=20.0, max_val=80.0)
        assert stats1 == stats2

    def test_inequality(self):
        stats1 = FittedStatistics(mean=50.0, std=10.0, min_val=20.0, max_val=80.0)
        stats2 = FittedStatistics(mean=51.0, std=10.0, min_val=20.0, max_val=80.0)
        assert stats1 != stats2

    def test_large_values(self):
        stats = FittedStatistics(mean=1e15, std=1e10, min_val=-1e15, max_val=1e15)
        assert stats.mean == 1e15
