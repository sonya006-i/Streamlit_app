"""
Preprocessing module for UNSW-NB15 Network Intrusion Detection System.
Provides feature definitions, ColumnTransformer construction, and robust input validation.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# 34 Input Features strictly defined from UNSW-NB15 schema
CATEGORICAL_FEATURES: List[str] = [
    "proto",
    "service",
    "state",
]

NUMERICAL_FEATURES: List[str] = [
    "dur",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "swin",
    "stcpb",
    "dtcpb",
    "dwin",
    "tcprtt",
    "synack",
    "ackdat",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "is_sm_ips_ports",
]

ALL_FEATURES: List[str] = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# Target column names
BINARY_TARGET: str = "label"
MULTICLASS_TARGET: str = "attack_cat"

# Common default fallback values for inference inputs
DEFAULT_FEATURE_VALUES: Dict[str, Any] = {
    "proto": "tcp",
    "service": "-",
    "state": "FIN",
    "dur": 0.05,
    "spkts": 10,
    "dpkts": 8,
    "sbytes": 500,
    "dbytes": 350,
    "rate": 150.0,
    "sload": 50000.0,
    "dload": 30000.0,
    "sloss": 0,
    "dloss": 0,
    "sinpkt": 5.0,
    "dinpkt": 5.0,
    "sjit": 1.0,
    "djit": 1.0,
    "swin": 255,
    "stcpb": 1000000,
    "dtcpb": 1000000,
    "dwin": 255,
    "tcprtt": 0.001,
    "synack": 0.0005,
    "ackdat": 0.0005,
    "smean": 50,
    "dmean": 45,
    "trans_depth": 0,
    "response_body_len": 0,
    "ct_src_dport_ltm": 1,
    "ct_dst_sport_ltm": 1,
    "is_ftp_login": 0,
    "ct_ftp_cmd": 0,
    "ct_flw_http_mthd": 0,
    "is_sm_ips_ports": 0,
}


def create_preprocessor() -> ColumnTransformer:
    """
    Creates a scikit-learn ColumnTransformer with OneHotEncoder for categoricals
    (ignoring unknown categories at test/inference time) and StandardScaler for numerics.
    """
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    numerical_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("num", numerical_transformer, NUMERICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans string categories (stripping whitespace, lowercasing proto/service if appropriate),
    and converts numeric types to float64/int64 safely.
    """
    df = df.copy()

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def validate_input_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Validates that the input dataframe has the required schema for inference.
    Fills missing features with safe defaults and drops unexpected extra columns.

    Returns:
        (sanitized_df, missing_cols, extra_cols)
    """
    df = clean_dataframe(df)
    existing_cols = set(df.columns)
    required_cols = set(ALL_FEATURES)

    missing_cols = list(required_cols - existing_cols)
    extra_cols = list(existing_cols - required_cols - {BINARY_TARGET, MULTICLASS_TARGET})

    sanitized = pd.DataFrame(index=df.index)

    for col in ALL_FEATURES:
        if col in df.columns:
            sanitized[col] = df[col]
        else:
            default_val = DEFAULT_FEATURE_VALUES.get(col, 0.0)
            sanitized[col] = default_val

    # Ensure correct column ordering
    sanitized = sanitized[ALL_FEATURES]

    return sanitized, missing_cols, extra_cols
