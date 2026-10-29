"""
Generates lightweight sample files for Streamlit Cloud deployment and demo testing.
- data/sample_eval.parquet (5,000 rows stratified test subset)
- data/sample_batch_demo.csv (100 rows formatted for Batch Analysis upload)
"""

from pathlib import Path
import pandas as pd
from src.data_loader import load_dataset
from src.preprocessing import ALL_FEATURES, BINARY_TARGET, MULTICLASS_TARGET

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def generate_samples():
    print("Generating sample evaluation parquet and batch demo CSV...")
    df_test = load_dataset("test")

    # 1. Stratified 5,000-sample parquet for fast in-app Traffic Explorer and Evaluation
    sample_eval = df_test.groupby("label", group_keys=False).apply(
        lambda x: x.sample(int(round(5000 * len(x) / len(df_test))), random_state=42)
    ).reset_index(drop=True)

    sample_eval_path = DATA_DIR / "sample_eval.parquet"
    sample_eval.to_parquet(sample_eval_path, index=False)
    print(f"Created: {sample_eval_path} ({len(sample_eval)} rows, {sample_eval_path.stat().st_size / 1024:.1f} KB)")

    # 2. Batch demo CSV with 100 rows containing diverse attacks and normal flows
    batch_demo = df_test.sample(100, random_state=101).copy()
    # Let's keep features and optionally include or omit targets to demonstrate validation
    batch_demo_csv_path = DATA_DIR / "sample_batch_demo.csv"
    batch_demo.to_csv(batch_demo_csv_path, index=False)
    print(f"Created: {batch_demo_csv_path} ({len(batch_demo)} rows, {batch_demo_csv_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    generate_samples()
