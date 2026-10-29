"""
Unit tests for pipeline predictions and batch inference.
"""

import pytest
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from src.preprocessing import ALL_FEATURES, DEFAULT_FEATURE_VALUES
from src.evaluation import run_batch_inference

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "intrusion_pipeline.joblib"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BATCH_DEMO_CSV = DATA_DIR / "sample_batch_demo.csv"


def test_model_artifact_exists():
    """Verify that the trained pipeline artifact exists."""
    assert MODEL_PATH.exists(), f"Trained model not found at {MODEL_PATH}"


def test_single_flow_prediction():
    """Verify pipeline prediction on a single flow."""
    pipeline = joblib.load(MODEL_PATH)
    sample_row = pd.DataFrame([{col: DEFAULT_FEATURE_VALUES[col] for col in ALL_FEATURES}])

    pred = pipeline.predict(sample_row)
    proba = pipeline.predict_proba(sample_row)

    assert pred[0] in [0, 1]
    assert 0.0 <= proba[0, 1] <= 1.0
    assert np.isclose(proba[0, 0] + proba[0, 1], 1.0)


def test_batch_inference_pipeline():
    """Verify batch inference pipeline produces expected verdicts and threat levels."""
    pipeline = joblib.load(MODEL_PATH)
    if BATCH_DEMO_CSV.exists():
        df_demo = pd.read_csv(BATCH_DEMO_CSV)
    else:
        df_demo = pd.DataFrame([{col: DEFAULT_FEATURE_VALUES[col] for col in ALL_FEATURES}])

    results = run_batch_inference(pipeline, None, df_demo)

    assert len(results) == len(df_demo)
    assert "risk_probability" in results.columns
    assert "predicted_label" in results.columns
    assert "verdict" in results.columns
    assert "threat_level" in results.columns

    for threat in results["threat_level"]:
        assert threat in ["CRITICAL RISK", "HIGH RISK", "ELEVATED RISK", "LOW RISK", "BENIGN / NORMAL"]
