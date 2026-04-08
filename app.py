import streamlit as st
import pandas as pd

from b import (
    add_proxy_qualified,
    compute_fairness_metrics,
    compute_odds_ratios,
    run_shap_analysis,
)

st.set_page_config(page_title="Hiring Bias Analysis", layout="wide")

st.title("Tech Hiring Bias Analysis Dashboard")
st.write(
    "Upload a CSV file → automatically compute Disparate Impact (DI), "
    "Equal Opportunity (EO), Odds Ratios, and SHAP explanations."
)

st.sidebar.header("Inputs & Parameters")

uploaded = st.sidebar.file_uploader(
    "Upload hiring data CSV", type=["csv"]
)

top_quantile = st.sidebar.slider(
    "Proxy Qualified: Select top X% as Qualified",
    min_value=0.10,
    max_value=0.60,
    value=0.40,
    step=0.05
)

baseline_group_default = "Male_White"

run_btn = st.sidebar.button("Run Analysis")

if uploaded is None:
    st.info("Please upload a CSV file on the left to begin.")
    st.stop()

df = pd.read_csv(uploaded)

if "Gender" not in df.columns or "Race" not in df.columns:
    st.error(
        "CSV must contain both 'Gender' and 'Race' columns "
        "(e.g. Male/Female, White/Black/Asian/Hispanic)."
    )
    st.stop()

df["Group"] = df["Gender"].astype(str) + "_" + df["Race"].astype(str)

baseline_group = st.sidebar.selectbox(
    "Select Baseline Group (all other groups are compared against it)",
    options=sorted(df["Group"].unique().tolist()),
    index=sorted(df["Group"].unique().tolist()).index(baseline_group_default)
    if baseline_group_default in df["Group"].unique()
    else 0
)

with st.expander("Data Preview (first 20 rows)", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)

if not run_btn:
    st.warning("Click **Run Analysis** in the sidebar to start.")
    st.stop()

df2 = add_proxy_qualified(df, top_quantile=top_quantile)

col1, col2 = st.columns(2)

with col1:
    st.subheader("PART 1: Fairness Metrics (DI / EO Gap / TPR)")
    fairness_df = compute_fairness_metrics(
        df2, baseline_group=baseline_group
    )
    st.dataframe(fairness_df.round(4), use_container_width=True)

    st.markdown("**Disparate Impact (DI) bar chart (closer to 1 is fairer)**")
    di_plot = (
        fairness_df[["Group", "DI"]]
        .set_index("Group")
        .sort_values("DI")
    )
    st.bar_chart(di_plot)

    di_viol = fairness_df[fairness_df["DI"] < 0.8]
    eo_viol = fairness_df[fairness_df["EO_Gap"].abs() > 0.10]

    if len(di_viol) > 0:
        st.error(
            " Groups with DI < 0.8:\n"
            + "\n".join(di_viol["Group"].tolist())
        )
    else:
        st.success(" No groups with DI < 0.8")

    if len(eo_viol) > 0:
        st.error(
            " Groups with |EO Gap| > 10%:\n"
            + "\n".join(eo_viol["Group"].tolist())
        )
    else:
        st.success(" No groups with |EO Gap| > 10%")

with col2:
    st.subheader(
        "PART 2: Odds Ratios "
        "(group disadvantage controlling for qualifications)"
    )
    odds_df = compute_odds_ratios(
        df2, baseline_group=baseline_group
    )

    group_or = odds_df[
        odds_df["Term"].astype(str).str.startswith("G_")
    ].copy()
    group_or["Group"] = group_or["Term"].str.replace(
        "G_", "", regex=False
    )
    group_or = group_or[
        ["Group", "Odds_Ratio", "Coefficient"]
    ].sort_values("Odds_Ratio")

    st.dataframe(group_or.round(4), use_container_width=True)

    st.markdown(
        "**Odds Ratio bar chart "
        "(OR < 0.8 is commonly considered a strong disadvantage)**"
    )
    or_plot = group_or.set_index("Group")[["Odds_Ratio"]]
    st.bar_chart(or_plot)

    or_viol = group_or[group_or["Odds_Ratio"] < 0.8]
    if len(or_viol) > 0:
        st.error(
            " Groups with OR < 0.8:\n"
            + "\n".join(or_viol["Group"].tolist())
        )
    else:
        st.success(" No groups with OR < 0.8")

st.subheader(
    "PART 3: SHAP "
    "(Which features contribute most to bias?)"
)
shap_results = run_shap_analysis(
    df2, baseline_group=baseline_group
)

if shap_results is None:
    st.info(
        "SHAP analysis not run "
        "(shap may not be installed). "
        "To enable: `pip install shap`"
    )
else:
    summary = shap_results["summary"].copy()
    st.dataframe(summary, use_container_width=True)

    st.markdown(
        "**Top bias-contributing features "
        "(most negative Avg_Diff first)**"
    )
    st.bar_chart(summary[["Avg_Diff"]])

st.subheader("Download Results")
c1, c2, c3 = st.columns(3)

with c1:
    st.download_button(
        "Download fairness_metrics.csv",
        fairness_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="fairness_metrics.csv",
        mime="text/csv",
    )

with c2:
    st.download_button(
        "Download odds_ratios.csv",
        odds_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="odds_ratios.csv",
        mime="text/csv",
    )

with c3:
    if shap_results is not None:
        st.download_button(
            "Download shap_feature_summary.csv",
            shap_results["summary"]
            .to_csv()
            .encode("utf-8-sig"),
            file_name="shap_feature_summary.csv",
            mime="text/csv",
        )
    else:
        st.caption(
            "SHAP was not run, so no SHAP output is available for download."
        )
