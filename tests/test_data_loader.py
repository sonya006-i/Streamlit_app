"""
Unit tests for data loader module.
"""

import pytest
import pandas as pd
from pathlib import Path
from src.data_loader import (
    get_dataset_paths,
    load_dataset,
    get_dataset_overview_stats,
    find_file,
)
from src.preprocessing import ALL_FEATURES, BINARY_TARGET


def test_find_dataset_or_sample():
    """Verify that either the full dataset or the sample parquet exists."""
    train_path, test_path = get_dataset_paths()
    sample_path = find_file(["sample_eval.parquet"])
    assert train_path is not None or sample_path is not None, "Neither full dataset nor sample file found"


def test_load_sample_dataset():
    """Verify loading sample dataset returns valid dataframe with expected schema."""
    df = load_dataset("sample", sample_size=200)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    # Check that key features exist
    for col in ["proto", "service", "state", "dur", "spkts", "sbytes"]:
        assert col in df.columns, f"Expected column {col} in dataset"


def test_dataset_overview_stats():
    """Verify statistical summary computation."""
    data = {
        "proto": ["tcp", "udp", "tcp"],
        "service": ["http", "dns", "ftp"],
        "state": ["FIN", "INT", "CON"],
        "label": [0, 1, 0],
    }
    df = pd.DataFrame(data)
    stats = get_dataset_overview_stats(df)
    assert stats["total_records"] == 3
    assert stats["attack_count"] == 1
    assert stats["normal_count"] == 2
    assert pytest.approx(stats["attack_pct"], 0.1) == 33.33
    assert stats["proto_count"] == 2
    assert stats["service_count"] == 3
