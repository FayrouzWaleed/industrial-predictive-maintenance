"""Industrial Predictive Maintenance dashboard (Streamlit).

Run with:  streamlit run app.py
"""
import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

import pm_core as pm
from pathlib import Path
import joblib


st.set_page_config(page_title="Predictive Maintenance", page_icon="🏭", layout="wide", initial_sidebar_state="expanded")

st.title("Industrial Predictive Maintenance Dashboard")
st.caption("AI4I 2020 dataset · failure-type classifier tuned to minimise false alarms (FDR)")

# ----------------------------------------------------------------------------
# Data + models (cached: trained once per session)
# ----------------------------------------------------------------------------
BUNDLE_PATH = Path(__file__).resolve().parent / "bundle.joblib"


@st.cache_resource(show_spinner="Loading models...")
def get_bundle(csv_bytes, csv_path):
    if csv_bytes is None and BUNDLE_PATH.exists():
        return joblib.load(BUNDLE_PATH)
    raw = pd.read_csv(io.BytesIO(csv_bytes)) if csv_bytes else pd.read_csv(csv_path)
    return pm.train_bundle(raw)


with st.sidebar:
    st.title("🏭 Settings")
    uploaded = st.file_uploader("Dataset (optional)", type="csv",
                                help="Leave empty to use ai4i2020.csv from the data folder.")
    path = pm.find_dataset()
    if uploaded is None and path is None:
        st.error("ai4i2020.csv not found. Put it in ../data/ or upload it above.")
        st.stop()

    bundle = get_bundle(uploaded.getvalue() if uploaded else None, str(path))
    classes, nf = bundle["class_names"], bundle["nf"]

    st.subheader("Model")
    model_name = st.radio("Classifier", list(bundle["models"]), horizontal=True)
    min_recall = st.slider("Minimum recall (FDR rule)", 0.50, 0.95, 0.70, 0.05)

    st.subheader("Costs (relative units, assumed)")
    c_fa = st.number_input("False alarm", 0.0, 1000.0, 1.0, 1.0)
    c_wt = st.number_input("Wrong failure type", 0.0, 1000.0, 2.0, 1.0)
    c_mf = st.number_input("Missed failure", 0.0, 1000.0, 10.0, 1.0)

    # Validation tables for the chosen model (thresholds are always chosen on validation)
    val_table = pm.threshold_table(bundle["y_val"], bundle["proba"][model_name]["val"], nf)
    val_cost = pm.add_cost_column(val_table, c_fa, c_wt, c_mf)
    fdr_row = pm.select_fdr_rule(val_table, min_recall)
    cost_row = pm.select_cost_rule(val_cost)

    st.subheader("Operating threshold")
    options = ["FDR rule", "Cost rule", "Manual"]
    mode = st.radio("Choose by", options, help=(
        "FDR rule: lowest false discovery rate with recall above the minimum (validation set).\n\n"
        "Cost rule: lowest total cost (validation set)."))
    if mode == "FDR rule":
        if fdr_row is None:
            st.warning("No threshold reaches that recall, using the one with the best recall.")
            threshold = float(val_table.sort_values("Recall", ascending=False).iloc[0]["Threshold"])
        else:
            threshold = float(fdr_row["Threshold"])
    elif mode == "Cost rule":
        threshold = float(cost_row["Threshold"])
    else:
        threshold = st.slider("Threshold", 0.05, 0.95, 0.50, 0.05)
    st.metric("Active threshold", f"{threshold:.2f}")


tabs = st.tabs(["📊 Overview", "🔗 Lead indicator", "🎯 Models & threshold",
                "💰 Cost-sensitive", "🚨 Anomaly detection", "🔮 Live prediction"])

df, info = bundle["df"], bundle["info"]
y_test, proba_test = bundle["y_test"], bundle["proba"][model_name]["test"]


def pct(x):
    return f"{x:.1%}"


# ----------------------------------------------------------------------------
# 1. Overview
# ----------------------------------------------------------------------------
with tabs[0]:
    failures = int((df["Failure Type"] != "No Failure").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Machines (after cleaning)", f"{info['clean_rows']:,}")
    c2.metric("Failures", f"{failures:,}", pct(failures / len(df)))
    c3.metric("Inconsistent rows removed", info["inconsistent_removed"])
    c4.metric("Random failures (RNF) removed", info["rnf_removed"])

    left, right = st.columns(2)
    counts = df["Failure Type"].value_counts().reset_index()
    counts.columns = ["Failure Type", "Count"]
    fig = px.bar(counts, x="Failure Type", y="Count", color="Failure Type", text="Count",
                 title="Class distribution (log scale)", log_y=True)
    fig.update_layout(showlegend=False)
    left.plotly_chart(fig, width="stretch")

    by_type = (df.assign(fail=df["Failure Type"] != "No Failure")
               .groupby("Type")["fail"].mean().reset_index())
    fig = px.bar(by_type, x="Type", y="fail", text=by_type["fail"].map(pct),
                 title="Failure rate by product quality type")
    fig.update_yaxes(tickformat=".0%", title="Failure rate")
    right.plotly_chart(fig, width="stretch")

    st.info(f"A model that always says 'No Failure' scores "
            f"{pct(1 - failures / len(df))} accuracy and catches nothing, so accuracy alone is misleading here.")

    sensor = st.selectbox("Sensor vs failure type", pm.SENSORS + pm.ENGINEERED)
    fig = px.box(df, x="Failure Type", y=sensor, color="Failure Type",
                 category_orders={"Failure Type": ["No Failure"] + list(pm.FAILURE_NAMES.values())})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------------------------
# 2. Lead indicator (bonus)
# ----------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Which sensor is the lead indicator of failure?")
    st.caption("Each failure type is compared with normal machines (correlation with a 0/1 flag). "
               "Raw sensors name the lead; engineered features are shown for comparison.")
    corr = bundle["corr"]
    lead_table, ranking = pm.lead_sensors(corr)

    fig = px.imshow(corr.round(3), text_auto=True, aspect="auto", zmin=-0.3, zmax=0.3,
                    color_continuous_scale="RdBu_r", title="Correlation with each failure type")
    st.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    a.markdown("**Lead sensor per failure type**")
    a.dataframe(lead_table.round(3), hide_index=True, width="stretch")
    rank_df = ranking.reset_index()
    rank_df.columns = ["Sensor", "Mean |correlation|"]
    fig = px.bar(rank_df, x="Mean |correlation|", y="Sensor", orientation="h",
                 title="Average strength across failure types")
    fig.update_yaxes(autorange="reversed")
    b.plotly_chart(fig, width="stretch")

    top = ranking.index[0]
    st.success(f"Overall lead indicator: **{top}** (mean |r| = {ranking.iloc[0]:.3f}), "
               f"followed by {ranking.index[1]} ({ranking.iloc[1]:.3f}).")
    st.warning("Correlation is not causation, and every raw sensor is weak on its own: failures come from "
               "combinations (e.g. small temperature gap together with low speed). The dataset has no "
               "time series, so 'lead' means strongest association, not 'changes first'.")

# ----------------------------------------------------------------------------
# 3. Models & threshold
# ----------------------------------------------------------------------------
with tabs[2]:
    st.subheader(f"{model_name} at threshold {threshold:.2f} ({mode})")
    st.caption("Thresholds are chosen on the validation set. The numbers below are on the untouched test set.")

    default_pred = proba_test.argmax(axis=1)
    tuned_pred = pm.predict_with_threshold(proba_test, threshold, nf)
    m_def = pm.failure_metrics(y_test, default_pred, nf)
    m_tun = pm.failure_metrics(y_test, tuned_pred, nf)

    k = st.columns(6)
    k[0].metric("FDR", pct(m_tun["FDR"]), f"{(m_tun['FDR'] - m_def['FDR']) * 100:+.1f} pts vs default",
                delta_color="inverse")
    k[1].metric("Precision", pct(m_tun["Precision"]))
    k[2].metric("Recall", pct(m_tun["Recall"]))
    k[3].metric("False alarms", m_tun["False Positives"], f"{m_tun['False Positives'] - m_def['False Positives']:+d}",
                delta_color="inverse")
    k[4].metric("Missed failures", m_tun["False Negatives"], f"{m_tun['False Negatives'] - m_def['False Negatives']:+d}",
                delta_color="inverse")
    k[5].metric("Accuracy", pct(m_tun["Accuracy"]), f"{(m_tun['Accuracy'] - m_def['Accuracy']) * 100:+.2f} pts")

    compare = pd.DataFrame({"Default (most likely class)": m_def, f"Tuned (threshold {threshold:.2f})": m_tun}).T
    compare = compare[["FDR", "Precision", "Recall", "F1-score", "False Positives",
                       "False Negatives", "Wrong Type", "Accuracy"]]
    st.dataframe(compare.style.format({c: "{:.3f}" for c in ["FDR", "Precision", "Recall", "F1-score", "Accuracy"]}),
                 width="stretch")

    left, right = st.columns(2)
    cm = confusion_matrix(y_test, tuned_pred, labels=range(len(classes)))
    fig = px.imshow(cm, x=classes, y=classes, text_auto=True, color_continuous_scale="Blues",
                    labels=dict(x="Predicted", y="Actual"), title="Confusion matrix (test)")
    left.plotly_chart(fig, width="stretch")

    fig = go.Figure()
    fig.add_scatter(x=val_table["Threshold"], y=val_table["FDR"], name="FDR", mode="lines+markers")
    fig.add_scatter(x=val_table["Threshold"], y=val_table["Recall"], name="Recall", mode="lines+markers")
    fig.add_vline(x=threshold, line_dash="dash", annotation_text="active")
    fig.add_hline(y=min_recall, line_dash="dot", line_color="gray", annotation_text="min recall")
    fig.update_layout(title="Validation: FDR and recall vs threshold", xaxis_title="Threshold", yaxis_tickformat=".0%")
    right.plotly_chart(fig, width="stretch")

    with st.expander("Validation threshold table"):
        st.dataframe(val_table.round(3), hide_index=True, width="stretch")
    with st.expander("Per-class report (test)"):
        rep = classification_report(y_test, tuned_pred, labels=range(len(classes)),
                                    target_names=classes, zero_division=0, output_dict=True)
        st.dataframe(pd.DataFrame(rep).T.round(3), width="stretch")
        tw = classes.index("Tool Wear") if "Tool Wear" in classes else None
        if tw is not None and (tuned_pred == tw).sum() == 0:
            st.warning("No Tool Wear alarms at this threshold: this failure happens at a random time between "
                       "200 and 240 min of wear, and there are very few examples.")

    st.markdown("**Both models at their own FDR-rule threshold (test set)**")
    rows = []
    for name in bundle["models"]:
        vt = pm.threshold_table(bundle["y_val"], bundle["proba"][name]["val"], nf)
        sel = pm.select_fdr_rule(vt, min_recall)
        if sel is None:
            continue
        p = pm.predict_with_threshold(bundle["proba"][name]["test"], sel["Threshold"], nf)
        m = pm.failure_metrics(y_test, p, nf)
        rows.append({"Model": name, "Threshold": sel["Threshold"], "FDR": m["FDR"], "Precision": m["Precision"],
                     "Recall": m["Recall"], "False Positives": m["False Positives"],
                     "False Negatives": m["False Negatives"], "Accuracy": m["Accuracy"]})
    if rows:
        st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")

# ----------------------------------------------------------------------------
# 4. Cost-sensitive
# ----------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Cost-sensitive threshold selection")
    st.caption("Costs come from the sidebar. They are assumptions: replace them with the factory's real numbers.")

    fig = go.Figure()
    for name in bundle["models"]:
        vt = pm.add_cost_column(
            pm.threshold_table(bundle["y_val"], bundle["proba"][name]["val"], nf), c_fa, c_wt, c_mf)
        fig.add_scatter(x=vt["Threshold"], y=vt["Total Cost"], name=name, mode="lines+markers")
    fig.update_layout(title="Total cost vs threshold (validation)", xaxis_title="Threshold",
                      yaxis_title="Total cost")
    st.plotly_chart(fig, width="stretch")

    rows = []
    for name in bundle["models"]:
        vt = pm.threshold_table(bundle["y_val"], bundle["proba"][name]["val"], nf)
        picks = {"FDR rule": pm.select_fdr_rule(vt, min_recall),
                 "Cost rule": pm.select_cost_rule(pm.add_cost_column(vt, c_fa, c_wt, c_mf))}
        for rule, sel in picks.items():
            if sel is None:
                continue
            p = pm.predict_with_threshold(bundle["proba"][name]["test"], sel["Threshold"], nf)
            m = pm.failure_metrics(y_test, p, nf)
            rows.append({"Model": name, "Rule": rule, "Threshold": sel["Threshold"],
                         "False Positives": m["False Positives"], "Wrong Type": m["Wrong Type"],
                         "False Negatives": m["False Negatives"], "Recall": m["Recall"], "FDR": m["FDR"],
                         "Total Cost": pm.total_cost(m, c_fa, c_wt, c_mf)})
    st.markdown("**Test set: FDR rule vs cost rule**")
    res = pd.DataFrame(rows)
    st.dataframe(res.round(3), hide_index=True, width="stretch")

    st.markdown("**How sensitive is the choice to the cost of a missed failure?**")
    ratios = [1, 2, 5, 10, 20, 50]
    rows = []
    for name in bundle["models"]:
        vt = pm.threshold_table(bundle["y_val"], bundle["proba"][name]["val"], nf)
        for r in ratios:
            sel = pm.select_cost_rule(pm.add_cost_column(vt, c_fa, c_wt, c_fa * r))
            p = pm.predict_with_threshold(bundle["proba"][name]["test"], sel["Threshold"], nf)
            m = pm.failure_metrics(y_test, p, nf)
            rows.append({"Model": name, "Missed / false-alarm cost": r, "Threshold": sel["Threshold"],
                         "False Positives": m["False Positives"], "False Negatives": m["False Negatives"],
                         "Test Total Cost": pm.total_cost(m, c_fa, c_wt, c_fa * r)})
    sens = pd.DataFrame(rows)
    fig = px.line(sens, x="Missed / false-alarm cost", y="Test Total Cost", color="Model", markers=True)
    st.plotly_chart(fig, width="stretch")
    with st.expander("Sensitivity table"):
        st.dataframe(sens.round(3), hide_index=True, width="stretch")

# ----------------------------------------------------------------------------
# 5. Anomaly detection
# ----------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Isolation Forest (trained on normal machines only)")
    yb = (y_test != nf).astype(int)
    hist = pd.DataFrame({"Anomaly score": bundle["anomaly"]["test"],
                         "Machine": np.where(yb == 1, "Failure", "Normal")})
    fig = px.histogram(hist, x="Anomaly score", color="Machine", histnorm="probability density",
                       barmode="overlay", nbins=40, opacity=0.6, title="Anomaly score on the test set")
    st.plotly_chart(fig, width="stretch")

    a_table, a_best = pm.anomaly_table(bundle, min_recall)
    a, b = st.columns(2)
    a.markdown("**Validation: alarm threshold as a percentile of normal machines**")
    a.dataframe(a_table.round(3), hide_index=True, width="stretch")
    if a_best is not None:
        res = pm.anomaly_test_result(bundle, a_best["Threshold"])
        b.markdown(f"**Test result at the {int(a_best['Percentile'])}th percentile**")
        b.metric("FDR", pct(res["FDR"]))
        b.metric("Recall", pct(res["Recall"]))
        b.metric("False alarms", res["False Positives"])
        st.error(f"About {res['FDR']:.0%} of its alarms are false: the anomaly detector alone cannot meet the "
                 "FDR requirement. It is useful as a label-free second opinion.")
    else:
        b.warning("No percentile reaches the minimum recall.")

    c, d = st.columns(2)
    c.markdown("**Ranking quality (threshold-free)**")
    c.dataframe(pm.ranking_comparison(bundle).round(3), hide_index=True, width="stretch")
    d.markdown("**Isolation Forest ROC-AUC by failure type**")
    d.dataframe(pm.anomaly_auc_by_type(bundle).round(3), hide_index=True, width="stretch")

# ----------------------------------------------------------------------------
# 6. Live prediction
# ----------------------------------------------------------------------------
PRESETS = {
    "Normal machine": dict(t="M", air=300.0, proc=310.0, rpm=1500, torque=40.0, wear=100),
    "Heat dissipation": dict(t="L", air=302.0, proc=310.5, rpm=1350, torque=45.0, wear=50),
    "Power failure": dict(t="M", air=300.0, proc=310.0, rpm=2600, torque=65.0, wear=80),
    "Overstrain": dict(t="L", air=300.0, proc=310.0, rpm=1300, torque=60.0, wear=190),
    "Tool wear": dict(t="L", air=300.0, proc=310.0, rpm=1500, torque=40.0, wear=225),
}
DEFAULTS = PRESETS["Normal machine"]
for key, value in DEFAULTS.items():
    st.session_state.setdefault(f"in_{key}", value)


def apply_preset(name):
    for key, value in PRESETS[name].items():
        st.session_state[f"in_{key}"] = value


with tabs[5]:
    st.subheader("Enter machine readings and get a prediction")
    st.caption(f"Using {model_name} with threshold {threshold:.2f} ({mode}) from the sidebar.")

    st.markdown("**Quick examples**")
    cols = st.columns(len(PRESETS))
    for col, name in zip(cols, PRESETS):
        col.button(name, on_click=apply_preset, args=(name,), width="stretch")

    c1, c2, c3 = st.columns(3)
    t_in = c1.selectbox("Product quality type", ["L", "M", "H"], key="in_t")
    air = c1.number_input("Air temperature [K]", 290.0, 310.0, step=0.1, key="in_air")
    proc = c2.number_input("Process temperature [K]", 300.0, 320.0, step=0.1, key="in_proc")
    rpm = c2.number_input("Rotational speed [rpm]", 1000, 3000, step=10, key="in_rpm")
    torque = c3.number_input("Torque [Nm]", 0.0, 100.0, step=0.5, key="in_torque")
    wear = c3.number_input("Tool wear [min]", 0, 300, step=1, key="in_wear")

    row = {"Type": t_in, "Air temperature [K]": air, "Process temperature [K]": proc,
           "Rotational speed [rpm]": rpm, "Torque [Nm]": torque, "Tool wear [min]": wear}
    X_in = pm.build_input_frame([row])
    proba, pred, labels, anomaly_score = pm.predict_frame(bundle, model_name, X_in, threshold)
    label, p_fail = labels[0], 1 - proba[0][nf]

    st.divider()
    if pred[0] == nf:
        st.success("### ✅ No alarm: machine looks healthy")
    else:
        st.error(f"### 🚨 Alarm: predicted **{label}**")

    r1, r2 = st.columns([1, 1])
    prob_df = pd.DataFrame({"Class": classes, "Probability": proba[0]})
    fig = px.bar(prob_df, x="Probability", y="Class", orientation="h", text=prob_df["Probability"].map(pct),
                 title="Class probabilities")
    fig.add_vline(x=threshold, line_dash="dash", annotation_text="alarm threshold")
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    r1.plotly_chart(fig, width="stretch")

    r2.metric("Total failure probability", pct(p_fail))
    a_thr = a_best["Threshold"] if a_best is not None else np.percentile(bundle["anomaly"]["val"], 95)
    r2.metric("Anomaly score (Isolation Forest)", f"{anomaly_score[0]:.3f}",
              "unusual operating point" if anomaly_score[0] >= a_thr else "typical operating point",
              delta_color="inverse" if anomaly_score[0] >= a_thr else "normal")
    feat = X_in.iloc[0]
    r2.markdown(f"Temp difference **{feat['Temp difference [K]']:.1f} K** · Power **{feat['Power [W]']:,.0f} W** · "
                f"Wear × Torque **{feat['Wear x Torque']:,.0f}**")

    st.markdown("**Physical rule check** (documented AI4I failure rules, independent of the model)")
    checks = pm.physical_checks(row)
    st.dataframe(checks.style.map(lambda v: "background-color: #f8d7da" if v is True else "", subset=["Triggered"]),
                 hide_index=True, width="stretch")
    if bool(checks["Triggered"].any()) != (pred[0] != nf):
        st.info("The model and the physical rules disagree. The model only alarms when its confidence passes "
                "the threshold, so borderline readings can differ.")

    with st.expander("📁 Batch prediction from a CSV"):
        need = ["Type"] + pm.SENSORS
        template = pd.DataFrame([row])[need]
        st.download_button("Download template", template.to_csv(index=False), "template.csv", "text/csv")
        batch = st.file_uploader("CSV with columns: " + ", ".join(need), type="csv", key="batch")
        if batch is not None:
            data = pd.read_csv(batch)
            missing = [c for c in need if c not in data.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                Xb = pm.build_input_frame(data[need].to_dict("records"))
                pb, pr, lb, an = pm.predict_frame(bundle, model_name, Xb, threshold)
                out = data.copy()
                out["Prediction"] = lb
                out["Failure probability"] = (1 - pb[:, nf]).round(3)
                out["Anomaly score"] = an.round(3)
                st.dataframe(out, hide_index=True, width="stretch")
                st.caption(f"{int((pr != nf).sum())} alarm(s) out of {len(out)} machines.")
                st.download_button("Download predictions", out.to_csv(index=False), "predictions.csv", "text/csv")
