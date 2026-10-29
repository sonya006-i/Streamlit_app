"""
Model training pipeline for UNSW-NB15 Network Intrusion Detection System.
Trains a reproducible scikit-learn Pipeline on official training data and evaluates on held-out test data.
Saves model artifacts and comprehensive evaluation metadata.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from src.preprocessing import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    BINARY_TARGET,
    MULTICLASS_TARGET,
    create_preprocessor,
    clean_dataframe,
)
from src.data_loader import load_dataset

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "intrusion_pipeline.joblib"
MULTICLASS_MODEL_PATH = MODELS_DIR / "multiclass_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"


def train_and_evaluate(
    sample_train_size: int = None,
    sample_test_size: int = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Loads official train/test sets, constructs Pipeline, fits on train only,
    and comprehensively evaluates on held-out test data.
    """
    print("=" * 60)
    print("◈ ALITA DEFENSIVE ML ENGINE: TRAINING PIPELINE ◈")
    print("=" * 60)

    # 1. Load official data
    print("\n[1/6] Loading official UNSW-NB15 dataset splits...")
    df_train = load_dataset("train", sample_size=sample_train_size, random_state=random_state)
    df_test = load_dataset("test", sample_size=sample_test_size, random_state=random_state)

    print(f"  Training samples: {len(df_train):,} rows")
    print(f"  Testing samples:  {len(df_test):,} rows")

    # 2. Separate features and target (strictly prevent leakage)
    print("\n[2/6] Preparing features and targets (leakage safeguards active)...")
    df_train_clean = clean_dataframe(df_train)
    df_test_clean = clean_dataframe(df_test)

    X_train = df_train_clean[ALL_FEATURES]
    y_train = df_train_clean[BINARY_TARGET].astype(int)
    y_train_multi = df_train_clean[MULTICLASS_TARGET].astype(str)

    X_test = df_test_clean[ALL_FEATURES]
    y_test = df_test_clean[BINARY_TARGET].astype(int)
    y_test_multi = df_test_clean[MULTICLASS_TARGET].astype(str)

    # Class distributions
    print(f"  Train binary balance: Normal={int((y_train == 0).sum())}, Attack={int((y_train == 1).sum())}")
    print(f"  Test binary balance:  Normal={int((y_test == 0).sum())}, Attack={int((y_test == 1).sum())}")

    # 3. Build scikit-learn Pipeline
    print("\n[3/6] Assembling scikit-learn Pipeline (ColumnTransformer + RandomForest)...")
    preprocessor = create_preprocessor()

    # Optimized for high predictive precision, low false-positive rate, and lightweight artifact size (<20MB)
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )

    binary_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    # 4. Fit Pipeline ONLY on training data
    print("\n[4/6] Fitting binary intrusion detection pipeline...")
    t0 = time.time()
    binary_pipeline.fit(X_train, y_train)
    fit_duration = time.time() - t0
    print(f"  Training completed in {fit_duration:.2f} seconds.")

    # 5. Evaluate on held-out test data
    print("\n[5/6] Evaluating on held-out test dataset (UNSW-NB15 Test Split)...")
    y_pred = binary_pipeline.predict(X_test)
    y_prob = binary_pipeline.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    cm = confusion_matrix(y_test, y_pred).tolist()
    cls_report = classification_report(y_test, y_pred, target_names=["Normal", "Attack"], output_dict=True)

    print(f"  Test Accuracy:  {acc * 100:.2f}%")
    print(f"  Test Precision: {prec * 100:.2f}%")
    print(f"  Test Recall:    {rec * 100:.2f}%")
    print(f"  Test F1-Score:  {f1 * 100:.2f}%")
    print(f"  Test ROC-AUC:   {roc_auc:.4f}")
    print(f"  Confusion Matrix (TN, FP / FN, TP): {cm}")

    # Extract Feature Importances
    preprocessor_fitted = binary_pipeline.named_steps["preprocessor"]
    rf_fitted = binary_pipeline.named_steps["classifier"]
    
    # Get feature names from ColumnTransformer
    try:
        encoded_feature_names = preprocessor_fitted.get_feature_names_out().tolist()
        importances = rf_fitted.feature_importances_.tolist()
        
        # Aggregate one-hot encoded features back to original feature names
        aggregated_importances: Dict[str, float] = {feat: 0.0 for feat in ALL_FEATURES}
        for name, imp in zip(encoded_feature_names, importances):
            # Check which original feature this encoded feature matches
            matched = False
            for orig in ALL_FEATURES:
                if name == orig or name.startswith(f"{orig}_"):
                    aggregated_importances[orig] += imp
                    matched = True
                    break
            if not matched:
                aggregated_importances[name] = imp
                
        # Sort by importance descending
        sorted_importances = sorted(aggregated_importances.items(), key=lambda x: x[1], reverse=True)
    except Exception as e:
        print(f"Warning: could not compute feature names out: {e}")
        sorted_importances = []

    # 6. Train Secondary Multiclass Classifier for Attack Categories
    print("\n[6/6] Training secondary multiclass attack classifier...")
    multi_preprocessor = create_preprocessor()
    multi_classifier = RandomForestClassifier(
        n_estimators=80,
        max_depth=15,
        min_samples_split=12,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    multi_pipeline = Pipeline([
        ("preprocessor", multi_preprocessor),
        ("classifier", multi_classifier),
    ])
    multi_pipeline.fit(X_train, y_train_multi)
    y_multi_pred = multi_pipeline.predict(X_test)
    multi_acc = float(accuracy_score(y_test_multi, y_multi_pred))
    print(f"  Multiclass Test Accuracy: {multi_acc * 100:.2f}%")

    # Serialize artifacts
    print(f"\nSaving binary pipeline to: {MODEL_PATH}")
    joblib.dump(binary_pipeline, MODEL_PATH, compress=3)
    
    print(f"Saving multiclass pipeline to: {MULTICLASS_MODEL_PATH}")
    joblib.dump(multi_pipeline, MULTICLASS_MODEL_PATH, compress=3)

    metadata: Dict[str, Any] = {
        "model_type": "RandomForestClassifier",
        "task": "Network Intrusion Detection",
        "training_samples": len(df_train),
        "test_samples": len(df_test),
        "fit_duration_seconds": round(fit_duration, 2),
        "evaluation_metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "multiclass_accuracy": round(multi_acc, 4),
            "confusion_matrix": cm,
            "classification_report": cls_report,
        },
        "top_features": sorted_importances[:20],
        "all_feature_names": ALL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 16,
            "min_samples_split": 10,
            "min_samples_leaf": 4,
            "class_weight": "balanced",
            "random_state": random_state,
        },
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to: {METADATA_PATH}")
    print("\n✓ MODEL TRAINING & EVALUATION COMPLETED WITH ZERO DATA LEAKAGE.")
    return metadata


if __name__ == "__main__":
    train_and_evaluate()
