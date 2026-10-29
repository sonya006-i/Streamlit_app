"""
Unit tests for preprocessing, schema validation, and ColumnTransformer pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    create_preprocessor,
    clean_dataframe,
    validate_input_features,
    DEFAULT_FEATURE_VALUES,
)


def test_feature_list_definitions():
    """Verify feature counts match the UNSW-NB15 schema."""
    assert len(CATEGORICAL_FEATURES) == 3
    assert len(NUMERICAL_FEATURES) == 31
    assert len(ALL_FEATURES) == 34


def test_clean_dataframe():
    """Verify that strings are stripped and numbers coerced properly."""
    raw = pd.DataFrame({
        "proto": ["  tcp  ", "udp "],
        "dur": ["0.15", "invalid"],
        "spkts": [10, 20],
    })
    cleaned = clean_dataframe(raw)
    assert cleaned["proto"].tolist() == ["tcp", "udp"]
    assert cleaned["dur"].iloc[0] == 0.15
    assert cleaned["dur"].iloc[1] == 0.0  # coerced to 0.0


def test_validate_input_features_handles_missing_and_extra():
    """Verify missing columns are filled with defaults and extra columns safely ignored."""
    partial_input = pd.DataFrame({
        "proto": ["tcp"],
        "dur": [0.05],
        "some_malicious_column": ["ignore_me"],
    })

    sanitized, missing_cols, extra_cols = validate_input_features(partial_input)
    assert len(sanitized) == 1
    assert list(sanitized.columns) == ALL_FEATURES
    assert "some_malicious_column" in extra_cols
    assert "spkts" in missing_cols
    assert sanitized["spkts"].iloc[0] == DEFAULT_FEATURE_VALUES["spkts"]


def test_preprocessor_handles_unseen_categories():
    """Verify that OneHotEncoder in preprocessor does not fail on unseen categories."""
    train_df = pd.DataFrame({col: [DEFAULT_FEATURE_VALUES[col]] for col in ALL_FEATURES})
    train_df["state"] = ["FIN"]
    train_df["proto"] = ["tcp"]
    train_df["service"] = ["http"]

    preprocessor = create_preprocessor()
    preprocessor.fit(train_df)

    # Test with unseen categories 'CLO' and 'rare_proto'
    unseen_df = train_df.copy()
    unseen_df["state"] = ["CLO"]
    unseen_df["proto"] = ["rare_unknown_proto"]

    # Should transform without raising an error due to handle_unknown='ignore'
    transformed = preprocessor.transform(unseen_df)
    assert transformed.shape[0] == 1
    assert not np.isnan(transformed).any()
