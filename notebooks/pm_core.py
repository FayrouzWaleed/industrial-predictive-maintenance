"""Core logic for the Predictive Maintenance dashboard.

Mirrors predictive_maintenance.ipynb: same cleaning, features, split, models,
FDR-based threshold selection, cost-sensitive selection and Isolation Forest.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
SENSORS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
ENGINEERED = ["Temp difference [K]", "Power [W]", "Wear x Torque"]
CATEGORICAL = ["Type"]
NUMERICAL = SENSORS + ENGINEERED
FEATURES = CATEGORICAL + NUMERICAL

ALL_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
MAIN_MODES = ["TWF", "HDF", "PWF", "OSF"]
FAILURE_NAMES = {
    "TWF": "Tool Wear",
    "HDF": "Heat Dissipation",
    "PWF": "Power Failure",
    "OSF": "Overstrain",
}
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
PERCENTILES = [50, 60, 70, 80, 85, 90, 95, 97, 98, 99]
DATA_CANDIDATES = ["../data/ai4i2020.csv", "data/ai4i2020.csv", "ai4i2020.csv"]


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def find_dataset():
    """Look for ai4i2020.csv in the usual places. Returns a Path or None."""
    here = Path(__file__).resolve().parent
    for candidate in DATA_CANDIDATES:
        for base in (Path.cwd(), here):
            path = (base / candidate).resolve()
            if path.exists():
                return path
    return None


def add_features(df):
    """Physics-based features (same formulas as the notebook)."""
    df = df.copy()
    df["Temp difference [K]"] = (
        df["Process temperature [K]"] - df["Air temperature [K]"]
    )
    df["Power [W]"] = (
        df["Torque [Nm]"] * df["Rotational speed [rpm]"] * 2 * np.pi / 60
    )
    df["Wear x Torque"] = df["Tool wear [min]"] * df["Torque [Nm]"]
    return df


def clean_and_label(raw):
    """Remove inconsistent + RNF rows and build the 5-class 'Failure Type'."""
    df = raw.copy()
    df["failure_mode_count"] = df[ALL_MODES].sum(axis=1)

    inconsistent = (df["Machine failure"] == 1) & (df["failure_mode_count"] == 0)
    n_inconsistent = int(inconsistent.sum())
    df = df[~inconsistent].copy()

    n_rnf = int((df["RNF"] == 1).sum())
    df = df[df["RNF"] == 0].copy()

    df["Failure Type"] = "No Failure"
    has_failure = df[MAIN_MODES].sum(axis=1) > 0
    df.loc[has_failure, "Failure Type"] = (
        df.loc[has_failure, MAIN_MODES].idxmax(axis=1).map(FAILURE_NAMES)
    )
    df = add_features(df)

    info = {
        "raw_rows": len(raw),
        "clean_rows": len(df),
        "inconsistent_removed": n_inconsistent,
        "rnf_removed": n_rnf,
        "multi_mode_rows": int((df["failure_mode_count"] > 1).sum()),
    }
    return df, info


def correlation_by_type(df):
    """Correlation of each feature with 'this failure type vs normal machines'."""
    columns = SENSORS + ENGINEERED
    out = pd.DataFrame(index=columns, columns=list(FAILURE_NAMES.values()), dtype=float)
    for failure_type in FAILURE_NAMES.values():
        subset = df[df["Failure Type"].isin(["No Failure", failure_type])]
        target = (subset["Failure Type"] == failure_type).astype(int)
        out[failure_type] = subset[columns].corrwith(target)
    return out


def lead_sensors(corr):
    """Strongest raw sensor per failure type + average ranking (raw sensors only)."""
    raw = corr.loc[SENSORS]
    rows = []
    for failure_type in raw.columns:
        best = raw[failure_type].abs().idxmax()
        rows.append({
            "Failure Type": failure_type,
            "Lead sensor": best,
            "Correlation": raw.loc[best, failure_type],
        })
    ranking = raw.abs().mean(axis=1).sort_values(ascending=False)
    return pd.DataFrame(rows), ranking


# ----------------------------------------------------------------------------
# Metrics (same definitions as the notebook)
# ----------------------------------------------------------------------------
def predict_with_threshold(proba, threshold, no_failure_index):
    """Alarm only if the best failure class has probability >= threshold."""
    pred = np.full(len(proba), no_failure_index)
    failure_proba = proba.copy()
    failure_proba[:, no_failure_index] = 0
    best_class = failure_proba.argmax(axis=1)
    best_score = failure_proba.max(axis=1)
    alarm = best_score >= threshold
    pred[alarm] = best_class[alarm]
    return pred


def failure_metrics(y_true, y_pred, no_failure_index):
    alarms = y_pred != no_failure_index
    actual = y_true != no_failure_index
    correct = alarms & (y_pred == y_true)

    fp = int((alarms & ~actual).sum())
    wrong_type = int((alarms & actual & (y_pred != y_true)).sum())
    fn = int((actual & ~alarms).sum())

    n_alarms = int(alarms.sum())
    n_correct = int(correct.sum())
    n_actual = int(actual.sum())

    precision = n_correct / n_alarms if n_alarms else 0.0
    recall = n_correct / n_actual if n_actual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fdr = (n_alarms - n_correct) / n_alarms if n_alarms else 0.0

    return {
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "False Positives": fp,
        "Wrong Type": wrong_type,
        "False Negatives": fn,
        "FDR": fdr,
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Alarms": n_alarms,
    }


def total_cost(m, c_false_alarm, c_wrong_type, c_missed):
    return (m["False Positives"] * c_false_alarm
            + m["Wrong Type"] * c_wrong_type
            + m["False Negatives"] * c_missed)


def threshold_table(y_true, proba, no_failure_index, thresholds=THRESHOLDS):
    rows = []
    for t in thresholds:
        pred = predict_with_threshold(proba, t, no_failure_index)
        m = failure_metrics(y_true, pred, no_failure_index)
        m["Threshold"] = t
        rows.append(m)
    cols = ["Threshold", "Precision", "Recall", "F1-score", "False Positives",
            "Wrong Type", "False Negatives", "FDR", "Accuracy"]
    return pd.DataFrame(rows)[cols]


def add_cost_column(table, c_false_alarm, c_wrong_type, c_missed):
    table = table.copy()
    table["Total Cost"] = (table["False Positives"] * c_false_alarm
                           + table["Wrong Type"] * c_wrong_type
                           + table["False Negatives"] * c_missed)
    return table


def select_fdr_rule(table, min_recall):
    """Lowest FDR with Recall >= min_recall (ties: fewer FP, higher recall)."""
    eligible = table[table["Recall"] >= min_recall]
    if eligible.empty:
        return None
    return eligible.sort_values(
        by=["FDR", "False Positives", "Recall"], ascending=[True, True, False]
    ).iloc[0]


def select_cost_rule(table_with_cost):
    """Lowest total cost (ties: fewer false positives)."""
    return table_with_cost.sort_values(
        by=["Total Cost", "False Positives"], ascending=[True, True]
    ).iloc[0]


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train_bundle(raw, seed=42):
    """Train everything the dashboard needs. Returns a dict (cached by the app)."""
    df, info = clean_and_label(raw)

    X = df[FEATURES].copy()
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["Failure Type"])
    class_names = list(encoder.classes_)
    nf = class_names.index("No Failure")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train)

    # Random Forest
    rf_pre = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("numerical", StandardScaler(), NUMERICAL),
    ])
    rf = Pipeline([
        ("preprocessor", rf_pre),
        ("classifier", RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=-1)),
    ])
    rf.fit(X_tr, y_tr)

    # XGBoost (class-balanced sample weights)
    xgb_pre = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("numerical", "passthrough", NUMERICAL),
    ])
    xgb = Pipeline([
        ("preprocessor", xgb_pre),
        ("classifier", XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            objective="multi:softprob", eval_metric="mlogloss",
            random_state=seed, n_jobs=-1)),
    ])
    weights = compute_sample_weight(class_weight="balanced", y=y_tr)
    xgb.fit(X_tr, y_tr, classifier__sample_weight=weights)

    models = {"Random Forest": rf, "XGBoost": xgb}
    proba = {
        name: {"val": m.predict_proba(X_val), "test": m.predict_proba(X_test)}
        for name, m in models.items()
    }

    # Isolation Forest: trained on normal machines only
    iso_pre = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("numerical", StandardScaler(), NUMERICAL),
    ])
    iso = Pipeline([
        ("preprocessor", iso_pre),
        ("detector", IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1)),
    ])
    iso.fit(X_tr[y_tr == nf])
    anomaly = {
        "val": -iso.score_samples(X_val),
        "test": -iso.score_samples(X_test),
    }

    corr = correlation_by_type(df)

    return {
        "df": df, "info": info, "class_names": class_names, "nf": nf,
        "X_val": X_val, "X_test": X_test, "y_val": y_val, "y_test": y_test,
        "models": models, "proba": proba, "iso": iso, "anomaly": anomaly,
        "corr": corr,
    }


# ----------------------------------------------------------------------------
# Anomaly detection helpers
# ----------------------------------------------------------------------------
def anomaly_table(bundle, min_recall):
    nf = bundle["nf"]
    yv = (bundle["y_val"] != nf).astype(int)
    score = bundle["anomaly"]["val"]
    rows = []
    for p in PERCENTILES:
        thr = np.percentile(score[yv == 0], p)
        pred = (score >= thr).astype(int)
        tp = int(((pred == 1) & (yv == 1)).sum())
        fp = int(((pred == 1) & (yv == 0)).sum())
        fn = int(((pred == 0) & (yv == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append({
            "Percentile": p, "Threshold": thr, "Precision": precision,
            "Recall": recall, "F1-score": f1, "False Positives": fp,
            "False Negatives": fn, "FDR": fp / (fp + tp) if (fp + tp) else 0.0,
        })
    table = pd.DataFrame(rows)
    eligible = table[table["Recall"] >= min_recall]
    best = None
    if not eligible.empty:
        best = eligible.sort_values(
            by=["FDR", "False Positives", "Recall"], ascending=[True, True, False]
        ).iloc[0]
    return table, best


def anomaly_test_result(bundle, threshold):
    nf = bundle["nf"]
    yt = (bundle["y_test"] != nf).astype(int)
    pred = (bundle["anomaly"]["test"] >= threshold).astype(int)
    tp = int(((pred == 1) & (yt == 1)).sum())
    fp = int(((pred == 1) & (yt == 0)).sum())
    fn = int(((pred == 0) & (yt == 1)).sum())
    return {
        "Precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "Recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "False Positives": fp, "False Negatives": fn,
        "FDR": fp / (fp + tp) if (fp + tp) else 0.0,
    }


def ranking_comparison(bundle):
    """Threshold-free comparison: ROC-AUC and Average Precision on the test set."""
    nf = bundle["nf"]
    yt = (bundle["y_test"] != nf).astype(int)
    scores = {"Isolation Forest": bundle["anomaly"]["test"]}
    for name in bundle["models"]:
        scores[name] = 1 - bundle["proba"][name]["test"][:, nf]
    rows = [{"Model": n, "ROC-AUC": roc_auc_score(yt, s),
             "Average Precision": average_precision_score(yt, s)}
            for n, s in scores.items()]
    return pd.DataFrame(rows)


def anomaly_auc_by_type(bundle):
    nf, y = bundle["nf"], bundle["y_test"]
    rows = []
    for idx, name in enumerate(bundle["class_names"]):
        if idx == nf:
            continue
        mask = (y == idx) | (y == nf)
        if (y == idx).sum() == 0:
            continue
        auc = roc_auc_score((y[mask] == idx).astype(int), bundle["anomaly"]["test"][mask])
        rows.append({"Failure Type": name, "ROC-AUC": auc})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Single / batch prediction
# ----------------------------------------------------------------------------
def build_input_frame(rows):
    """rows: list of dicts with Type + the 5 raw sensors -> model-ready frame."""
    return add_features(pd.DataFrame(rows))[FEATURES]


def predict_frame(bundle, model_name, X, threshold):
    nf = bundle["nf"]
    proba = bundle["models"][model_name].predict_proba(X)
    pred = predict_with_threshold(proba, threshold, nf)
    labels = [bundle["class_names"][i] for i in pred]
    anomaly_score = -bundle["iso"].score_samples(X)
    return proba, pred, labels, anomaly_score


def physical_checks(row):
    """Documented AI4I 2020 failure rules, evaluated for one machine (sanity check)."""
    wear, torque = row["Tool wear [min]"], row["Torque [Nm]"]
    rpm = row["Rotational speed [rpm]"]
    diff = row["Process temperature [K]"] - row["Air temperature [K]"]
    power = torque * rpm * 2 * np.pi / 60
    strain_limit = {"L": 11000, "M": 12000, "H": 13000}[row["Type"]]
    return pd.DataFrame([
        {"Failure mode": "Tool Wear",
         "Rule": "Tool wear between 200 and 240 min",
         "Current value": f"{wear:.0f} min", "Triggered": 200 <= wear <= 240},
        {"Failure mode": "Heat Dissipation",
         "Rule": "Temp difference < 8.6 K and speed < 1380 rpm",
         "Current value": f"{diff:.1f} K, {rpm:.0f} rpm",
         "Triggered": bool(diff < 8.6 and rpm < 1380)},
        {"Failure mode": "Power Failure",
         "Rule": "Power < 3,500 W or > 9,000 W",
         "Current value": f"{power:,.0f} W", "Triggered": bool(power < 3500 or power > 9000)},
        {"Failure mode": "Overstrain",
         "Rule": f"Wear x Torque > {strain_limit:,} (type {row['Type']})",
         "Current value": f"{wear * torque:,.0f}", "Triggered": bool(wear * torque > strain_limit)},
    ])
