# -*- coding: utf-8 -*-
"""
=============================================================================
NEXUS-ABI | Layer 2: Predictive Core
File: train_churn.py
=============================================================================

PURPOSE:
  Trains two XGBoost models on the customer dataset:
    1. Churn Classifier  → predicts IF a customer will churn (probability 0-1)
    2. CLV Regressor     → predicts HOW MUCH a customer is worth (in $)

WHY XGBOOST AND NOT AN LLM?
  An LLM asked "Will customer X churn?" will produce a confident-sounding
  but numerically meaningless answer. XGBoost learns from 10,000 historical
  examples and produces a calibrated probability. This is the "numerical
  ground truth" that the Strategist Agent in Layer 4 will cite in its answers.

  Think of it this way:
    XGBoost → "73.4% churn probability"          (precise, numerical)
    Llama 3 → "Based on this risk, you should..." (contextual, strategic)
  Together they are Hybrid Intelligence.

FEATURE ENGINEERING:
  Raw columns aren't fed directly to XGBoost. We create derived features
  that capture business insight — these are the signals the model learns from.

  Raw column            → Engineered feature
  ───────────────────────────────────────────
  monthly_spend         → spend_per_user (spend efficiency)
  tenure_months         → is_new_customer (< 6 months = higher risk)
  support_tickets_count → ticket_rate (tickets per month of tenure)
  last_login_days_ago   → is_inactive (> 30 days = danger signal)
  plan_type             → plan_encoded (Starter=0, Growth=1, Enterprise=2, Legacy=3)

OUTPUTS:
  02_predictive_core/models/
    ├── churn_model.pkl      → Saved XGBoost churn classifier
    ├── clv_model.pkl        → Saved XGBoost CLV regressor
    └── feature_names.json   → Column list (needed for agent inference)

RUN:
  python 02_predictive_core/train_churn.py
=============================================================================
"""

import json
import warnings
from pathlib import Path

import io
import sys
import joblib

# Force UTF-8 output on Windows terminals to avoid encoding crashes
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
DATA_PATH  = Path("01_data_pipeline/raw/customers.csv")
MODEL_DIR  = Path("02_predictive_core/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CHURN_MODEL_PATH   = MODEL_DIR / "churn_model.pkl"
CLV_MODEL_PATH     = MODEL_DIR / "clv_model.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.json"

RANDOM_SEED = 42
console = Console()


# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================

def load_data() -> pd.DataFrame:
    """Loads customer CSV and does a basic sanity check."""
    logger.info("Loading data from {}", DATA_PATH)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"customers.csv not found at {DATA_PATH}.\n"
            "Run: python 01_data_pipeline/synthetic_gen.py first."
        )

    df = pd.read_csv(DATA_PATH)
    logger.success("Loaded {:,} rows × {} columns", len(df), len(df.columns))

    # Quick sanity check
    assert "churned" in df.columns, "Missing 'churned' target column!"
    assert "clv"     in df.columns, "Missing 'clv' target column!"
    assert df["churned"].isin([0, 1]).all(), "churned column must be binary (0/1)!"

    return df


# =============================================================================
# STEP 2: FEATURE ENGINEERING
# =============================================================================

def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Transforms raw columns into ML-ready features.

    This is the most important step — raw data rarely speaks for itself.
    Each feature below encodes a BUSINESS INSIGHT into a number.
    """
    logger.info("Engineering features...")

    fe = pd.DataFrame()

    # --- DIRECT NUMERIC FEATURES ---
    # These are already numbers, just need to be passed through cleanly
    fe["tenure_months"]         = df["tenure_months"].clip(lower=1)
    fe["monthly_spend"]         = df["monthly_spend"].clip(lower=0)
    fe["support_tickets_count"] = df["support_tickets_count"].clip(lower=0)
    fe["last_login_days_ago"]   = df["last_login_days_ago"].clip(lower=0)
    fe["num_users"]             = df["num_users"].clip(lower=1)

    # --- DERIVED FEATURES (Business Intelligence encoded as math) ---

    # Spend per user: low spend-per-user = underutilising = churn risk
    fe["spend_per_user"] = (df["monthly_spend"] / df["num_users"].clip(lower=1)).round(2)

    # Ticket rate: tickets per month — a rising rate signals escalating problems
    fe["ticket_rate"] = (
        df["support_tickets_count"] / df["tenure_months"].clip(lower=1)
    ).round(4)

    # Is new customer: <6 months → haven't seen full product value yet
    fe["is_new_customer"] = (df["tenure_months"] < 6).astype(int)

    # Is inactive: not logged in for 30+ days → disengaged
    fe["is_inactive"] = (df["last_login_days_ago"] > 30).astype(int)

    # Is Legacy plan: highest churn risk plan
    fe["is_legacy_plan"] = (df["plan_type"] == "Legacy").astype(int)

    # Revenue risk score: compound signal (spend × churn indicators)
    fe["revenue_risk"] = (
        fe["monthly_spend"] * (fe["is_inactive"] + fe["ticket_rate"] + fe["is_new_customer"])
    ).round(2)

    # --- CATEGORICAL ENCODING ---
    # XGBoost needs numbers, not strings. We use LabelEncoder per column.

    plan_order = {"Starter": 0, "Growth": 1, "Enterprise": 2, "Legacy": 3}
    fe["plan_encoded"] = df["plan_type"].map(plan_order).fillna(0).astype(int)

    industry_encoder = LabelEncoder()
    fe["industry_encoded"] = industry_encoder.fit_transform(df["industry"].fillna("Unknown"))

    region_encoder = LabelEncoder()
    fe["region_encoded"] = region_encoder.fit_transform(df["region"].fillna("Unknown"))

    feature_names = fe.columns.tolist()
    logger.success("Engineered {} features: {}", len(feature_names), feature_names)
    return fe, feature_names


# =============================================================================
# STEP 3: TRAIN CHURN CLASSIFIER
# =============================================================================

def train_churn_model(X_train, X_test, y_train, y_test) -> XGBClassifier:
    """
    Trains an XGBoost binary classifier to predict churn (0 or 1).

    KEY HYPERPARAMETERS EXPLAINED:
      n_estimators=300    → Number of trees. More = better, but diminishing returns.
      max_depth=5         → How deep each tree can go. 5 is safe to avoid overfitting.
      learning_rate=0.05  → Small steps = more robust generalisation.
      scale_pos_weight    → Handles class imbalance. If 20% churn, weight churners
                            4x more so the model pays attention to them.
      eval_metric='auc'   → We optimise for AUC, not accuracy.
                            Why? A model that says "no churn" for everyone gets
                            80% accuracy but is completely useless for the business.
    """
    logger.info("Training XGBoost Churn Classifier...")

    # Handle class imbalance: weight churned=1 class higher
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale = round(n_neg / n_pos, 2)
    logger.info("Class balance — Not churned: {:,} | Churned: {:,} | Scale weight: {}",
                n_neg, n_pos, scale)

    model = XGBClassifier(
        n_estimators      = 300,
        max_depth         = 5,
        learning_rate     = 0.05,
        subsample         = 0.8,        # Use 80% of rows per tree (prevents overfitting)
        colsample_bytree  = 0.8,        # Use 80% of features per tree
        scale_pos_weight  = scale,      # Fix class imbalance
        use_label_encoder = False,
        eval_metric       = "auc",
        random_state      = RANDOM_SEED,
        n_jobs            = -1,         # Use all CPU cores
        verbosity         = 0,
    )

    model.fit(
        X_train, y_train,
        eval_set              = [(X_test, y_test)],
        verbose               = False,
    )

    # Cross-validation to confirm we're not just overfitting to the test split
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    logger.info("5-Fold CV AUC: {:.4f} ± {:.4f}", cv_scores.mean(), cv_scores.std())

    return model


def evaluate_churn_model(model: XGBClassifier, X_test, y_test) -> dict:
    """Evaluates the churn model and returns metrics as a dict."""
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_pred_prob), 4),
        "precision": round(float(classification_report(y_test, y_pred, output_dict=True)["1"]["precision"]), 4),
        "recall":    round(float(classification_report(y_test, y_pred, output_dict=True)["1"]["recall"]), 4),
        "f1":        round(float(classification_report(y_test, y_pred, output_dict=True)["1"]["f1-score"]), 4),
    }
    return metrics


# =============================================================================
# STEP 4: TRAIN CLV REGRESSOR
# =============================================================================

def train_clv_model(X_train, X_test, y_train, y_test) -> XGBRegressor:
    """
    Trains an XGBoost regressor to predict Customer Lifetime Value (CLV) in $.

    CLV = how much revenue a customer will generate over their relationship.
    This is used by the Strategist Agent to prioritise retention efforts:
    "Focus on customer X — they have a 70% churn risk AND a $24,000 CLV."
    """
    logger.info("Training XGBoost CLV Regressor...")

    model = XGBRegressor(
        n_estimators     = 300,
        max_depth        = 5,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        random_state     = RANDOM_SEED,
        n_jobs           = -1,
        verbosity        = 0,
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    return model


def evaluate_clv_model(model: XGBRegressor, X_test, y_test) -> dict:
    """Evaluates the CLV model and returns metrics."""
    y_pred = model.predict(X_test)
    metrics = {
        "mae":  round(mean_absolute_error(y_test, y_pred), 2),
        "r2":   round(r2_score(y_test, y_pred), 4),
        "rmse": round(float(np.sqrt(np.mean((y_test - y_pred) ** 2))), 2),
    }
    return metrics


# =============================================================================
# STEP 5: FEATURE IMPORTANCE — What did the model actually learn?
# =============================================================================

def print_feature_importance(model, feature_names: list[str], title: str) -> None:
    """
    Shows which features the model found most useful.
    This is the 'explainability' layer — we can tell stakeholders WHY
    the model predicts churn for a specific customer.
    """
    importance = model.feature_importances_
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
    fi_df = fi_df.sort_values("Importance", ascending=False).reset_index(drop=True)

    table = Table(title=f"[bold cyan]{title}[/bold cyan] — Top Feature Importances",
                  show_header=True, header_style="bold magenta")
    table.add_column("Rank", style="dim", width=5)
    table.add_column("Feature", width=25)
    table.add_column("Importance", justify="right")
    table.add_column("Bar", width=30)

    for i, row in fi_df.head(10).iterrows():
        bar_len = int(row["Importance"] * 100)
        bar     = "█" * bar_len + "░" * (30 - min(bar_len, 30))
        table.add_row(
            str(i + 1),
            row["Feature"],
            f"{row['Importance']:.4f}",
            f"[green]{bar}[/green]"
        )

    console.print(table)
    console.print()


# =============================================================================
# MAIN RUNNER
# =============================================================================

def main():
    console.print(Panel.fit(
        "[bold cyan]NEXUS-ABI[/bold cyan] | [white]Predictive Core Training[/white]\n"
        "[dim]XGBoost Churn Classifier + CLV Regressor[/dim]",
        border_style="cyan"
    ))

    # --- Load ---
    df = load_data()

    # --- Feature engineering ---
    X, feature_names = engineer_features(df)
    y_churn = df["churned"]
    y_clv   = df["clv"]

    # --- Train/test split ---
    # Stratified split ensures equal churn ratio in train and test sets
    X_train, X_test, y_churn_train, y_churn_test = train_test_split(
        X, y_churn, test_size=0.2, stratify=y_churn, random_state=RANDOM_SEED
    )
    _, _, y_clv_train, y_clv_test = train_test_split(
        X, y_clv, test_size=0.2, random_state=RANDOM_SEED
    )
    logger.info("Train: {:,} | Test: {:,}", len(X_train), len(X_test))

    # =========================================================================
    # TRAIN & EVALUATE: CHURN MODEL
    # =========================================================================
    churn_model   = train_churn_model(X_train, X_test, y_churn_train, y_churn_test)
    churn_metrics = evaluate_churn_model(churn_model, X_test, y_churn_test)

    console.print("\n[bold green]-- Churn Model Results --[/bold green]")
    churn_table = Table(show_header=True, header_style="bold magenta")
    churn_table.add_column("Metric",    style="dim", width=15)
    churn_table.add_column("Value",     justify="right", width=15)
    churn_table.add_column("Target",    justify="right", width=15)
    churn_table.add_column("Status",    width=10)

    targets = {"accuracy": 0.75, "roc_auc": 0.80, "precision": 0.65, "recall": 0.60, "f1": 0.65}
    for metric, value in churn_metrics.items():
        target = targets.get(metric, 0.70)
        status = "[green]✓ PASS[/green]" if value >= target else "[red]✗ FAIL[/red]"
        churn_table.add_row(metric, str(value), str(target), status)

    console.print(churn_table)
    print_feature_importance(churn_model, feature_names, "Churn Classifier")

    # =========================================================================
    # TRAIN & EVALUATE: CLV MODEL
    # =========================================================================
    clv_model   = train_clv_model(X_train, X_test, y_clv_train, y_clv_test)
    clv_metrics = evaluate_clv_model(clv_model, X_test, y_clv_test)

    console.print("[bold green]-- CLV Regressor Results --[/bold green]")
    clv_table = Table(show_header=True, header_style="bold magenta")
    clv_table.add_column("Metric", style="dim", width=15)
    clv_table.add_column("Value",  justify="right", width=15)

    for metric, value in clv_metrics.items():
        clv_table.add_row(metric, str(value))

    console.print(clv_table)
    print_feature_importance(clv_model, feature_names, "CLV Regressor")

    # =========================================================================
    # SAVE MODELS
    # =========================================================================
    logger.info("Saving models to {}", MODEL_DIR)

    joblib.dump(churn_model, CHURN_MODEL_PATH)
    logger.success("Saved → {}", CHURN_MODEL_PATH)

    joblib.dump(clv_model, CLV_MODEL_PATH)
    logger.success("Saved → {}", CLV_MODEL_PATH)

    # Save feature names so the agent knows what columns to pass at inference
    with open(FEATURE_NAMES_PATH, "w") as f:
        json.dump(feature_names, f, indent=2)
    logger.success("Saved → {}", FEATURE_NAMES_PATH)

    # =========================================================================
    # QUICK INFERENCE TEST — Prove the model works end-to-end
    # =========================================================================
    console.print("\n[bold yellow]── Quick Inference Test ──[/bold yellow]")
    console.print("[dim]Simulating agent calling the model for a high-risk customer...[/dim]\n")

    # A customer with all churn signals maxed out
    test_customer = pd.DataFrame([{
        "tenure_months":         2,      # Very new
        "monthly_spend":         99.0,   # Cheapest plan
        "support_tickets_count": 8,      # Lots of complaints
        "last_login_days_ago":   45,     # Inactive for 45 days
        "num_users":             1,
        "spend_per_user":        99.0,
        "ticket_rate":           4.0,    # 4 tickets per month
        "is_new_customer":       1,
        "is_inactive":           1,
        "is_legacy_plan":        1,
        "revenue_risk":          99.0 * (1 + 4.0 + 1),
        "plan_encoded":          3,      # Legacy
        "industry_encoded":      0,
        "region_encoded":        0,
    }])

    churn_prob = churn_model.predict_proba(test_customer)[0][1]
    clv_pred   = clv_model.predict(test_customer)[0]

    console.print(f"  [red]Churn Probability: {churn_prob*100:.1f}%[/red]")
    console.print(f"  [yellow]Predicted CLV: ${clv_pred:,.2f}[/yellow]")
    console.print(f"  [dim]Agent interpretation: {'HIGH RISK — Immediate retention action needed.' if churn_prob > 0.6 else 'MODERATE RISK — Monitor closely.'}[/dim]")

    console.print(f"\n[bold green]✓ Training complete. Models saved to {MODEL_DIR.resolve()}[/bold green]\n")


if __name__ == "__main__":
    main()
