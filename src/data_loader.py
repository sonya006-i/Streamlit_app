"""
Data loader module for UNSW-NB15 dataset.
Handles loading parquet/csv files, caching for Streamlit, schema validation, and sample generation.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
import pandas as pd
import numpy as np

# Primary candidate directories
DEFAULT_DATA_DIRS = [
    Path(os.environ["UNSW_DATA_DIR"]) if "UNSW_DATA_DIR" in os.environ else None,
    Path("~/UNSW-NB15").expanduser(),
    Path(__file__).resolve().parent.parent / "data",
    Path("./data"),
]
DEFAULT_DATA_DIRS = [d for d in DEFAULT_DATA_DIRS if d is not None]

TRAIN_FILENAMES = ["UNSW_NB15_training-set.parquet", "UNSW_NB15_training-set.csv"]
TEST_FILENAMES = ["UNSW_NB15_testing-set.parquet", "UNSW_NB15_testing-set.csv"]
SAMPLE_EVAL_FILENAME = "sample_eval.parquet"


def find_file(filenames: list, candidate_dirs: list = DEFAULT_DATA_DIRS) -> Optional[Path]:
    """Finds the first matching file across candidate directories."""
    for d in candidate_dirs:
        if d.exists():
            for fn in filenames:
                candidate = d / fn
                if candidate.exists() and candidate.is_file():
                    return candidate
    return None


def get_dataset_paths() -> Tuple[Optional[Path], Optional[Path]]:
    """Returns (train_path, test_path) if available."""
    train_path = find_file(TRAIN_FILENAMES)
    test_path = find_file(TEST_FILENAMES)
    return train_path, test_path


def load_dataframe_from_path(path: Path, sample_size: Optional[int] = None, random_state: int = 42) -> pd.DataFrame:
    """Reads Parquet or CSV safely, with optional stratified/random sampling."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if sample_size and len(df) > sample_size:
        if "label" in df.columns:
            # Stratified sample
            df = df.groupby("label", group_keys=False).apply(
                lambda x: x.sample(
                    int(np.rint(sample_size * len(x) / len(df))),
                    random_state=random_state
                )
            ).reset_index(drop=True)
        else:
            df = df.sample(sample_size, random_state=random_state).reset_index(drop=True)

    return df


def load_dataset(
    split: str = "train",
    sample_size: Optional[int] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Loads 'train', 'test', or 'sample' dataset.
    Falls back gracefully to bundled sample if full dataset is not on disk.
    """
    train_path, test_path = get_dataset_paths()

    if split == "train":
        if train_path:
            return load_dataframe_from_path(train_path, sample_size=sample_size, random_state=random_state)
        raise FileNotFoundError("Training dataset file not located.")

    elif split == "test":
        if test_path:
            return load_dataframe_from_path(test_path, sample_size=sample_size, random_state=random_state)
        # Check for fallback sample_eval.parquet
        sample_path = find_file([SAMPLE_EVAL_FILENAME])
        if sample_path:
            return load_dataframe_from_path(sample_path, sample_size=sample_size, random_state=random_state)
        raise FileNotFoundError("Testing dataset file not located.")

    elif split == "sample":
        sample_path = find_file([SAMPLE_EVAL_FILENAME])
        if sample_path:
            return load_dataframe_from_path(sample_path, sample_size=sample_size, random_state=random_state)
        if test_path:
            return load_dataframe_from_path(test_path, sample_size=sample_size or 5000, random_state=random_state)
        if train_path:
            return load_dataframe_from_path(train_path, sample_size=sample_size or 5000, random_state=random_state)
        raise FileNotFoundError("No dataset or sample file found.")

    else:
        raise ValueError(f"Unknown split: {split}. Choose 'train', 'test', or 'sample'.")


def get_dataset_overview_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Generates high-level statistical summary for UI KPI cards."""
    total_records = len(df)
    if "label" in df.columns:
        attack_count = int((df["label"] == 1).sum())
        normal_count = int((df["label"] == 0).sum())
        attack_pct = (attack_count / total_records) * 100 if total_records > 0 else 0.0
    else:
        attack_count = 0
        normal_count = total_records
        attack_pct = 0.0

    proto_count = int(df["proto"].nunique()) if "proto" in df.columns else 0
    service_count = int(df["service"].nunique()) if "service" in df.columns else 0

    return {
        "total_records": total_records,
        "attack_count": attack_count,
        "normal_count": normal_count,
        "attack_pct": attack_pct,
        "proto_count": proto_count,
        "service_count": service_count,
    }
