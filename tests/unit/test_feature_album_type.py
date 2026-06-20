"""Tests for AlbumTypeBlock feature computation.

Tests verify correct one-hot encoding of album types including:
- Vocabulary learning from training data
- One-hot encoding correctness
- Missing value handling (defaults to Album)
- Unknown type handling (all zeros)
- Frozen vocabulary immutability
"""

from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from panelcast.features.album_type import AlbumTypeBlock
from panelcast.features.base import FeatureContext, FeatureOutput
from panelcast.features.errors import NotFittedError


@pytest.fixture
def ctx():
    """Create a test FeatureContext."""
    return FeatureContext(config={}, random_state=42)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame with required columns."""
    return pd.DataFrame(
        {
            "Album_Type": ["Album", "EP", "Mixtape", "Compilation"],
        }
    )


class TestFitTransformEnforcement:
    """Tests for fit/transform state enforcement."""

    def test_transform_before_fit_raises_error(self, sample_df, ctx):
        """Transform without fit must raise NotFittedError."""
        block = AlbumTypeBlock()

        with pytest.raises(NotFittedError) as exc_info:
            block.transform(sample_df, ctx)

        assert "album_type" in str(exc_info.value)
        assert "has not been fitted yet" in str(exc_info.value)

    def test_is_fitted_false_initially(self):
        """is_fitted should be False before fit() is called."""
        block = AlbumTypeBlock()
        assert block.is_fitted is False

    def test_fit_sets_fitted_true(self, sample_df, ctx):
        """is_fitted should be True after fit() is called."""
        block = AlbumTypeBlock()
        block.fit(sample_df, ctx)
        assert block.is_fitted is True

    def test_fit_returns_self(self, sample_df, ctx):
        """fit() should return self for method chaining."""
        block = AlbumTypeBlock()
        result = block.fit(sample_df, ctx)
        assert result is block


class TestVocabularyLearning:
    """Tests for vocabulary learning from training data."""

    def test_fit_learns_vocabulary(self, ctx):
        """fit() should learn vocabulary from training data."""
        df = pd.DataFrame({"Album_Type": ["Album", "EP", "Album", "Mixtape"]})

        block = AlbumTypeBlock()
        block.fit(df, ctx)

        # Vocabulary should contain unique types (sorted)
        assert block.vocabulary.categories == ("Album", "EP", "Mixtape")

    def test_fit_learns_vocabulary_sorted(self, ctx):
        """Vocabulary should be sorted alphabetically."""
        df = pd.DataFrame({"Album_Type": ["Mixtape", "Album", "EP"]})

        block = AlbumTypeBlock()
        block.fit(df, ctx)

        # Should be sorted
        assert block.vocabulary.categories == ("Album", "EP", "Mixtape")

    def test_fit_ignores_null_in_vocabulary(self, ctx):
        """Null values should not appear in vocabulary."""
        df = pd.DataFrame({"Album_Type": ["Album", None, "EP"]})

        block = AlbumTypeBlock()
        block.fit(df, ctx)

        assert block.vocabulary.categories == ("Album", "EP")


class TestOneHotEncoding:
    """Tests for one-hot encoding correctness."""

    def test_one_hot_encoding_correct(self, ctx):
        """One-hot encoding should produce correct binary columns."""
        df = pd.DataFrame({"Album_Type": ["Album", "EP"]})

        block = AlbumTypeBlock()
        output = block.fit_transform(df, ctx)

        # Album row: is_album=1, is_ep=0
        assert output.data.loc[0, "is_album"] == 1
        assert output.data.loc[0, "is_ep"] == 0

        # EP row: is_album=0, is_ep=1
        assert output.data.loc[1, "is_album"] == 0
        assert output.data.loc[1, "is_ep"] == 1

    def test_one_hot_all_types(self, sample_df, ctx):
        """All album types should be correctly one-hot encoded."""
        block = AlbumTypeBlock()
        output = block.fit_transform(sample_df, ctx)

        # Row 0: Album
        assert output.data.loc[0, "is_album"] == 1
        assert output.data.loc[0, "is_ep"] == 0
        assert output.data.loc[0, "is_mixtape"] == 0
        assert output.data.loc[0, "is_compilation"] == 0

        # Row 1: EP
        assert output.data.loc[1, "is_album"] == 0
        assert output.data.loc[1, "is_ep"] == 1
        assert output.data.loc[1, "is_mixtape"] == 0
        assert output.data.loc[1, "is_compilation"] == 0

        # Row 2: Mixtape
        assert output.data.loc[2, "is_album"] == 0
        assert output.data.loc[2, "is_ep"] == 0
        assert output.data.loc[2, "is_mixtape"] == 1
        assert output.data.loc[2, "is_compilation"] == 0

        # Row 3: Compilation
        assert output.data.loc[3, "is_album"] == 0
        assert output.data.loc[3, "is_ep"] == 0
        assert output.data.loc[3, "is_mixtape"] == 0
        assert output.data.loc[3, "is_compilation"] == 1


class TestMissingValues:
    """Tests for missing Album_Type handling."""

    def test_missing_album_type_defaults_to_album(self, ctx):
        """Missing Album_Type should default to 'Album'."""
        train_df = pd.DataFrame({"Album_Type": ["Album", "EP"]})
        test_df = pd.DataFrame({"Album_Type": [None]})

        block = AlbumTypeBlock()
        block.fit(train_df, ctx)
        output = block.transform(test_df, ctx)

        # Null should be treated as Album
        assert output.data.loc[0, "is_album"] == 1
        assert output.data.loc[0, "is_ep"] == 0


class TestUnknownTypes:
    """Tests for unknown type handling."""

    def test_unknown_type_all_zeros(self, ctx):
        """Unknown types (not in training) should get all zeros."""
        train_df = pd.DataFrame({"Album_Type": ["Album", "EP"]})
        test_df = pd.DataFrame({"Album_Type": ["Unknown_Type"]})

        block = AlbumTypeBlock()
        block.fit(train_df, ctx)
        output = block.transform(test_df, ctx)

        # Unknown type should have all zeros
        assert output.data.loc[0, "is_album"] == 0
        assert output.data.loc[0, "is_ep"] == 0


class TestVocabularyImmutability:
    """Tests for frozen vocabulary."""

    def test_vocabulary_is_frozen(self, sample_df, ctx):
        """Attempt to modify vocabulary should raise FrozenInstanceError."""
        block = AlbumTypeBlock()
        block.fit(sample_df, ctx)

        with pytest.raises(FrozenInstanceError):
            block.vocabulary.categories = ("X", "Y", "Z")


class TestFeatureOutput:
    """Tests for FeatureOutput structure."""

    def test_output_is_feature_output(self, sample_df, ctx):
        """Output should be FeatureOutput instance."""
        block = AlbumTypeBlock()
        output = block.fit_transform(sample_df, ctx)
        assert isinstance(output, FeatureOutput)

    def test_output_has_correct_column_names(self, sample_df, ctx):
        """Output columns should be is_{type} format."""
        block = AlbumTypeBlock()
        output = block.fit_transform(sample_df, ctx)

        # Column names should be lowercase with is_ prefix
        expected_cols = ["is_album", "is_compilation", "is_ep", "is_mixtape"]
        assert list(output.data.columns) == expected_cols
        assert output.feature_names == expected_cols

    def test_output_preserves_original_index(self, ctx):
        """Output should have same index as input DataFrame."""
        df = pd.DataFrame(
            {"Album_Type": ["Album", "EP"]},
            index=[100, 200],
        )

        block = AlbumTypeBlock()
        output = block.fit_transform(df, ctx)

        assert output.data.index.tolist() == [100, 200]

    def test_output_has_metadata(self, sample_df, ctx):
        """Output should include block metadata."""
        block = AlbumTypeBlock()
        output = block.fit_transform(sample_df, ctx)

        assert "block" in output.metadata
        assert output.metadata["block"] == "album_type"
        assert "vocabulary_size" in output.metadata
        assert output.metadata["vocabulary_size"] == 4


class TestMissingColumns:
    """Tests for missing column validation."""

    def test_fit_raises_on_missing_columns(self, ctx):
        """fit() should raise ValueError if Album_Type column missing."""
        df = pd.DataFrame({"Other_Column": ["A", "B"]})

        block = AlbumTypeBlock()
        with pytest.raises(ValueError) as exc_info:
            block.fit(df, ctx)

        assert "missing required columns" in str(exc_info.value).lower()


class TestVocabularyAccess:
    """Tests for vocabulary property access."""

    def test_vocabulary_access_before_fit_raises(self):
        """Accessing vocabulary before fit should raise NotFittedError."""
        block = AlbumTypeBlock()

        with pytest.raises(NotFittedError):
            _ = block.vocabulary


# ---------------------------------------------------------------------------
# Additional album type tests
# ---------------------------------------------------------------------------


class TestAlbumTypeBlockAttributes:
    """Tests for block attributes and initialization."""

    def test_name_is_album_type(self):
        block = AlbumTypeBlock()
        assert block.name == "album_type"

    def test_requires_is_empty(self):
        block = AlbumTypeBlock()
        assert block.requires == []

    def test_required_columns_has_album_type(self):
        block = AlbumTypeBlock()
        assert "Album_Type" in block.required_columns

    def test_default_params_empty(self):
        block = AlbumTypeBlock()
        assert block.params == {}


class TestAlbumTypeVocabularyDetails:
    """Tests for vocabulary internal details."""

    def test_vocabulary_category_to_idx_maps_correctly(self, ctx):
        df = pd.DataFrame({"Album_Type": ["EP", "Album"]})
        block = AlbumTypeBlock()
        block.fit(df, ctx)
        vocab = block.vocabulary
        assert vocab.category_to_idx["Album"] == 0
        assert vocab.category_to_idx["EP"] == 1

    def test_vocabulary_unknown_idx_set(self, ctx):
        df = pd.DataFrame({"Album_Type": ["Album", "EP"]})
        block = AlbumTypeBlock()
        block.fit(df, ctx)
        vocab = block.vocabulary
        assert vocab.unknown_idx == len(vocab.categories)

    def test_vocabulary_single_category(self, ctx):
        df = pd.DataFrame({"Album_Type": ["Album", "Album", "Album"]})
        block = AlbumTypeBlock()
        block.fit(df, ctx)
        assert block.vocabulary.categories == ("Album",)


class TestAlbumTypeTransformDetails:
    """Tests for specific transform behaviors."""

    def test_transform_with_different_split(self, ctx):
        train_df = pd.DataFrame({"Album_Type": ["Album", "EP", "Mixtape"]})
        test_df = pd.DataFrame({"Album_Type": ["EP", "Album"]}, index=[10, 20])
        block = AlbumTypeBlock()
        block.fit(train_df, ctx)
        output = block.transform(test_df, ctx)
        # EP at index 10: is_album=0, is_ep=1
        assert output.data.loc[10, "is_ep"] == 1
        assert output.data.loc[10, "is_album"] == 0
        # Album at index 20: is_album=1
        assert output.data.loc[20, "is_album"] == 1

    def test_transform_all_unknown(self, ctx):
        train_df = pd.DataFrame({"Album_Type": ["Album"]})
        test_df = pd.DataFrame({"Album_Type": ["Unknown1", "Unknown2"]})
        block = AlbumTypeBlock()
        block.fit(train_df, ctx)
        output = block.transform(test_df, ctx)
        # All should be zeros
        assert output.data.loc[0, "is_album"] == 0
        assert output.data.loc[1, "is_album"] == 0

    def test_transform_multiple_nulls(self, ctx):
        train_df = pd.DataFrame({"Album_Type": ["Album", "EP"]})
        test_df = pd.DataFrame({"Album_Type": [None, None, None]})
        block = AlbumTypeBlock()
        block.fit(train_df, ctx)
        output = block.transform(test_df, ctx)
        # All nulls default to Album
        assert output.data["is_album"].sum() == 3

    def test_output_row_count_matches_input(self, sample_df, ctx):
        block = AlbumTypeBlock()
        output = block.fit_transform(sample_df, ctx)
        assert len(output.data) == len(sample_df)


class TestAlbumTypeFitTransformEquivalence:
    """Tests for equivalence of fit_transform vs fit+transform."""

    def test_fit_transform_equals_separate(self, sample_df, ctx):
        block1 = AlbumTypeBlock()
        output1 = block1.fit_transform(sample_df, ctx)

        block2 = AlbumTypeBlock()
        block2.fit(sample_df, ctx)
        output2 = block2.transform(sample_df, ctx)

        pd.testing.assert_frame_equal(output1.data, output2.data)

    def test_can_refit(self, ctx):
        df1 = pd.DataFrame({"Album_Type": ["Album", "EP"]})
        df2 = pd.DataFrame({"Album_Type": ["Mixtape", "Compilation"]})
        block = AlbumTypeBlock()
        block.fit(df1, ctx)
        assert block.vocabulary.categories == ("Album", "EP")
        block.fit(df2, ctx)
        assert block.vocabulary.categories == ("Compilation", "Mixtape")
