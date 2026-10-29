"""
Evaluation and visualization module for UNSW-NB15 Intrusion Detection System.
Provides Plotly visualizations tailored to the dark cyberpunk aesthetic and inference utilities.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.preprocessing import ALL_FEATURES, validate_input_features

# Dark Cyberpunk Plotly Palette
CYBER_BG = "#0B0F19"
CYBER_CARD_BG = "#131824"
CYBER_BORDER = "#1E2638"
CYBER_PINK = "#FF1493"
CYBER_NEON_GREEN = "#00FF66"
CYBER_CYAN = "#00E5FF"
CYBER_YELLOW = "#FFD700"
CYBER_RED = "#FF0055"
CYBER_TEXT = "#E6EDF3"
CYBER_MUTED = "#8B949E"


def create_cyber_layout(title: str, height: int = 400) -> Dict[str, Any]:
    """Returns standard Plotly layout options matching dark cybersecurity theme."""
    return {
        "title": {
            "text": f"<b>{title.upper()}</b>",
            "font": {"color": CYBER_TEXT, "size": 15, "family": "Inter, Roboto, sans-serif"},
            "x": 0.02,
        },
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(11,15,25,0.7)",
        "height": height,
        "margin": {"l": 50, "r": 30, "t": 60, "b": 40},
        "font": {"color": CYBER_TEXT, "family": "Inter, Roboto, sans-serif"},
        "xaxis": {
            "gridcolor": "#1C2436",
            "zerolinecolor": "#2A364F",
            "tickfont": {"color": CYBER_MUTED, "size": 11},
        },
        "yaxis": {
            "gridcolor": "#1C2436",
            "zerolinecolor": "#2A364F",
            "tickfont": {"color": CYBER_MUTED, "size": 11},
        },
    }


def plot_confusion_matrix(cm: List[List[int]], class_names: List[str] = ["Normal (0)", "Attack (1)"]) -> go.Figure:
    """Renders high-visibility confusion matrix heatmap."""
    z = np.array(cm)
    total = np.sum(z)
    
    annotations = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = z[i][j]
            pct = (val / total) * 100
            
            # Label distinction
            if i == 0 and j == 0:
                tag = "True Normal (TN)"
                color = CYBER_NEON_GREEN
            elif i == 1 and j == 1:
                tag = "True Intrusion (TP)"
                color = CYBER_NEON_GREEN
            elif i == 0 and j == 1:
                tag = "False Alarm (FP)"
                color = CYBER_YELLOW
            else:
                tag = "Missed Attack (FN)"
                color = CYBER_RED

            text = f"<b>{val:,}</b><br><span style='font-size:11px;color:{color}'>{tag}</span><br><span style='font-size:10px;color:{CYBER_MUTED}'>({pct:.1f}%)</span>"
            annotations.append(
                dict(
                    x=class_names[j],
                    y=class_names[i],
                    text=text,
                    showarrow=False,
                    font=dict(color=CYBER_TEXT, size=13),
                )
            )

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=class_names,
            y=class_names,
            colorscale=[
                [0.0, "#0E1524"],
                [0.5, "#271B38"],
                [1.0, "#FF1493"],
            ],
            showscale=False,
        )
    )

    layout = create_cyber_layout("Confusion Matrix Heatmap (Held-Out Test Set)", height=380)
    layout["annotations"] = annotations
    layout["xaxis"]["title"] = "<b>Predicted Label</b>"
    layout["yaxis"]["title"] = "<b>Actual Ground Truth</b>"
    fig.update_layout(**layout)
    return fig


def plot_roc_curve(fpr: List[float], tpr: List[float], roc_auc: float) -> go.Figure:
    """Renders glowing neon ROC Curve."""
    fig = go.Figure()

    # Baseline chance
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color=CYBER_MUTED, dash="dash", width=1.5),
            name="Random Baseline (AUC = 0.50)",
        )
    )

    # ROC Line
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            line=dict(color=CYBER_NEON_GREEN, width=3),
            name=f"Random Forest (AUC = {roc_auc:.4f})",
            fill="tozeroy",
            fillcolor="rgba(0, 255, 102, 0.08)",
        )
    )

    layout = create_cyber_layout(f"Receiver Operating Characteristic (ROC-AUC = {roc_auc:.4f})", height=380)
    layout["xaxis"]["title"] = "False Positive Rate (FPR)"
    layout["yaxis"]["title"] = "True Positive Rate (TPR / Recall)"
    layout["xaxis"]["range"] = [-0.01, 1.01]
    layout["yaxis"]["range"] = [-0.01, 1.03]
    fig.update_layout(**layout)
    return fig


def plot_feature_importances(top_features: List[Tuple[str, float]], max_display: int = 15) -> go.Figure:
    """Renders horizontal bar chart of top predictive features."""
    items = top_features[:max_display][::-1]
    features = [x[0] for x in items]
    scores = [x[1] * 100 for x in items]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=features,
            orientation="h",
            marker=dict(
                color=scores,
                colorscale=[
                    [0.0, "#00E5FF"],
                    [0.5, "#FF1493"],
                    [1.0, "#FF007F"],
                ],
                line=dict(color="#FF69B4", width=1),
            ),
            text=[f"{s:.2f}%" for s in scores],
            textposition="outside",
            textfont=dict(color=CYBER_TEXT, size=11),
        )
    )

    layout = create_cyber_layout("Top Predictive Traffic Features (Random Forest MDI)", height=460)
    layout["xaxis"]["title"] = "Relative Feature Importance (%)"
    layout["yaxis"]["title"] = "Flow Feature"
    fig.update_layout(**layout)
    return fig


def plot_attack_categories_distribution(df: pd.DataFrame) -> go.Figure:
    """Plots attack category breakdown in the traffic sample."""
    if "attack_cat" not in df.columns:
        return go.Figure()

    counts = df["attack_cat"].value_counts().reset_index()
    counts.columns = ["Category", "Count"]

    # Highlight Normal vs Attack with distinct cyber colors
    colors = [
        CYBER_NEON_GREEN if cat == "Normal" else CYBER_PINK
        for cat in counts["Category"]
    ]

    fig = go.Figure(
        go.Bar(
            x=counts["Category"],
            y=counts["Count"],
            marker=dict(color=colors, line=dict(color="#FFFFFF", width=0.5)),
            text=[f"{c:,}" for c in counts["Count"]],
            textposition="outside",
            textfont=dict(color=CYBER_TEXT, size=11),
        )
    )

    layout = create_cyber_layout("Traffic Volume by Category (Normal vs Attack Vectors)", height=400)
    layout["xaxis"]["title"] = "Attack Category"
    layout["yaxis"]["title"] = "Flow Records"
    fig.update_layout(**layout)
    return fig


def plot_traffic_scatter(df: pd.DataFrame, x_col: str, y_col: str) -> go.Figure:
    """Interactive scatter plot comparing two numerical features colored by label."""
    sample = df.sample(min(len(df), 2000), random_state=42) if len(df) > 2000 else df

    sample = sample.copy()
    if "label" in sample.columns:
        sample["Traffic Type"] = sample["label"].map({0: "Normal", 1: "Intrusion Attack"})
        color_discrete_map = {"Normal": CYBER_NEON_GREEN, "Intrusion Attack": CYBER_PINK}
        fig = px.scatter(
            sample,
            x=x_col,
            y=y_col,
            color="Traffic Type",
            color_discrete_map=color_discrete_map,
            hover_data=["proto", "service", "state"] if "proto" in sample.columns else None,
            opacity=0.7,
        )
    else:
        fig = px.scatter(sample, x=x_col, y=y_col, opacity=0.7)

    layout = create_cyber_layout(f"Feature Interaction: {x_col} vs {y_col}", height=420)
    fig.update_layout(**layout)
    return fig


def run_batch_inference(pipeline, multi_pipeline, df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Executes full batch inference with risk category calculation and multiclass classification.
    """
    sanitized, missing, extra = validate_input_features(df_input)

    # Binary predictions and probabilities
    probs = pipeline.predict_proba(sanitized)[:, 1]
    preds = (probs >= 0.5).astype(int)

    # Risk tiers
    def categorize_risk(p: float) -> str:
        if p >= 0.85:
            return "CRITICAL RISK"
        elif p >= 0.60:
            return "HIGH RISK"
        elif p >= 0.40:
            return "ELEVATED RISK"
        elif p >= 0.15:
            return "LOW RISK"
        else:
            return "BENIGN / NORMAL"

    risk_tiers = [categorize_risk(p) for p in probs]

    results = df_input.copy()
    results["risk_probability"] = np.round(probs, 4)
    results["predicted_label"] = preds
    results["verdict"] = ["ATTACK DETECTED" if p == 1 else "NORMAL" for p in preds]
    results["threat_level"] = risk_tiers

    # Multiclass predictions if available
    if multi_pipeline is not None:
        try:
            multi_preds = multi_pipeline.predict(sanitized)
            results["predicted_category"] = multi_preds
        except Exception:
            pass

    return results
