import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. Run: pip install shap")



def load_data(csv_path="tech_diversity_hiring_data.csv"):
    df = pd.read_csv(csv_path)
    df["Group"] = df["Gender"] + "_" + df["Race"]
    
    print("=" * 70)
    print("STEP 2: BIAS ANALYSIS")
    print("=" * 70)
    print(f"Loaded: {csv_path}")
    print(f"Samples: {len(df):,}")
    print(f"Groups: {sorted(df['Group'].unique())}")
    
    return df


def add_proxy_qualified(df, top_quantile=0.40):
    df = df.copy()
    
    score = (
        0.15 * zscore(df["YearsExperience"]) +
        0.15 * zscore(df["EducationLevel"]) +
        0.35 * zscore(df["AlgorithmSkill"]) +
        0.35 * zscore(df["OverallInterviewScore"])
    )
    
    df["QualificationScore"] = score
    cutoff = score.quantile(1 - top_quantile)
    df["Qualified"] = (score >= cutoff).astype(int)
    
    print(f"\nProxy Qualified: Top {top_quantile:.0%} = {df['Qualified'].sum():,} candidates")
    
    return df


def zscore(s):
    std = s.std()
    if std == 0:
        return s * 0
    return (s - s.mean()) / std


def compute_fairness_metrics(df, baseline_group="Male_White"):

    
    print("\n" + "=" * 70)
    print("PART 1: FAIRNESS METRICS")
    print("=" * 70)
    
    results = []
    
    for group in df["Group"].unique():
        mask = df["Group"] == group
        group_df = df[mask]
        qualified_df = group_df[group_df["Qualified"] == 1]
        
        results.append({
            "Group": group,
            "Size": len(group_df),
            "Hire_Rate": group_df["Hired"].mean(),
            "Qualified_Count": len(qualified_df),
            "TPR": qualified_df["Hired"].mean() if len(qualified_df) > 0 else np.nan
        })
    
    results_df = pd.DataFrame(results)
    
    baseline = results_df[results_df["Group"] == baseline_group].iloc[0]
    results_df["DI"] = results_df["Hire_Rate"] / baseline["Hire_Rate"]
    results_df["EO_Gap"] = results_df["TPR"] - baseline["TPR"]
    
    results_df = results_df.sort_values("DI", ascending=True)
    
    print(f"\nBaseline group: {baseline_group}")
    print(f"Baseline hire rate: {baseline['Hire_Rate']:.1%}")
    print("\n" + results_df.round(4).to_string(index=False))
    
    print("\n" + "-" * 50)
    print("VIOLATIONS:")
    print("-" * 50)
    
    di_violations = results_df[results_df["DI"] < 0.8]
    eo_violations = results_df[results_df["EO_Gap"].abs() > 0.10]
    
    if len(di_violations) > 0:
        print(f"\n  DI < 0.8 (potential discrimination):")
        for _, row in di_violations.iterrows():
            print(f"   {row['Group']}: DI = {row['DI']:.4f}")
    
    if len(eo_violations) > 0:
        print(f"\n |EO Gap| > 10% (unequal opportunity):")
        for _, row in eo_violations.iterrows():
            direction = "lower" if row["EO_Gap"] < 0 else "higher"
            print(f"   {row['Group']}: {abs(row['EO_Gap']):.1%} {direction} than baseline")
    
    return results_df


def compute_odds_ratios(df, baseline_group="Male_White"):

    print("\n" + "=" * 70)
    print("PART 2: ODDS RATIOS (controlling for qualifications)")
    print("=" * 70)
    
    feature_cols = ["YearsExperience", "EducationLevel", "AlgorithmSkill", "OverallInterviewScore"]

    group_dummies = pd.get_dummies(df["Group"], prefix="G")
    baseline_col = f"G_{baseline_group}"
    if baseline_col in group_dummies.columns:
        group_dummies = group_dummies.drop(columns=[baseline_col])
    

    scaler = StandardScaler()
    X_features = scaler.fit_transform(df[feature_cols])
    X = np.hstack([X_features, group_dummies.values])
    y = df["Hired"].values
    

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    feature_names = feature_cols + list(group_dummies.columns)
    
    results = []
    for i, name in enumerate(feature_names):
        coef = model.coef_[0][i]
        odds_ratio = np.exp(coef)
        results.append({
            "Term": name,
            "Coefficient": coef,
            "Odds_Ratio": odds_ratio
        })
    
    results_df = pd.DataFrame(results).sort_values("Odds_Ratio")
    
    print(f"\nBaseline group: {baseline_group}")
    print(f"Model accuracy: {model.score(X, y):.4f}")
    print("\nGroup odds ratios (OR < 1 = disadvantaged):")
    
    group_results = results_df[results_df["Term"].str.startswith("G_")]
    print(group_results.round(4).to_string(index=False))
    
    disadvantaged = group_results[group_results["Odds_Ratio"] < 0.8]
    if len(disadvantaged) > 0:
        print("\n  OR < 0.8 (significant disadvantage after controlling for qualifications):")
        for _, row in disadvantaged.iterrows():
            group_name = row["Term"].replace("G_", "")
            print(f"   {group_name}: OR = {row['Odds_Ratio']:.4f}")
    
    return results_df


def run_shap_analysis(df, baseline_group="Male_White"):
    if not SHAP_AVAILABLE:
        print("\n  SHAP not installed. Skipping SHAP analysis.")
        print("   Install with: pip install shap")
        return None
    
    print("\n" + "=" * 70)
    print("PART 3: SHAP ANALYSIS (Why is each group disadvantaged?)")
    print("=" * 70)
    
    feature_cols = [
        "YearsExperience", "EducationLevel", "AlgorithmSkill",
        "SystemDesignSkill", "OverallInterviewScore", "GitHubScore",
        "NumLanguages", "HasReferral", "ResumeScore", "TechInterviewScore",
        "CultureFitScore"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    print(f"Features: {feature_cols}")
    
    X = df[feature_cols].values
    y = df["Hired"].values
    groups = df["Group"].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_scaled, y)
    print(f"Model accuracy: {model.score(X_scaled, y):.4f}")
    
    print("\nCalculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
    
    baseline_mask = groups == baseline_group
    baseline_shap = shap_values[baseline_mask].mean(axis=0)
    
    all_results = []
    
    print("\n" + "-" * 70)
    print("WHY IS EACH GROUP DISADVANTAGED?")
    print("-" * 70)
    
    for group in sorted(df["Group"].unique()):
        if group == baseline_group:
            continue
        
        group_mask = groups == group
        group_shap = shap_values[group_mask].mean(axis=0)
        shap_diff = group_shap - baseline_shap
        
        group_hire_rate = df[group_mask]["Hired"].mean()
        baseline_hire_rate = df[baseline_mask]["Hired"].mean()
        hire_gap = group_hire_rate - baseline_hire_rate
        
        print(f"\n{group} (hire rate: {group_hire_rate:.1%}, gap: {hire_gap:+.1%}):")
        
        feature_diffs = list(zip(feature_cols, shap_diff))
        feature_diffs.sort(key=lambda x: x[1])
        
        disadvantages = [(f, d) for f, d in feature_diffs if d < -0.01]
        
        if disadvantages:
            print("  Main bias sources:")
            for feat, diff in disadvantages[:3]: 
                print(f"    • {feat}: {diff:+.4f}")
        else:
            print("  No single dominant bias source")
        
        for feat, diff in feature_diffs:
            all_results.append({
                "Group": group,
                "Feature": feat,
                "SHAP_Diff": diff,
                "Hire_Gap": hire_gap
            })
    
    results_df = pd.DataFrame(all_results)
    
    print("\n" + "-" * 70)
    print("TOP BIAS-CAUSING FEATURES (averaged across groups):")
    print("-" * 70)
    
    summary = results_df.groupby("Feature")["SHAP_Diff"].agg(["mean", "min"]).round(4)
    summary.columns = ["Avg_Diff", "Worst_Diff"]
    summary = summary.sort_values("Avg_Diff")
    print(summary.to_string())
    
    results_df.to_csv("shap_bias_analysis.csv", index=False)
    summary.to_csv("shap_feature_summary.csv")
    
    return {
        "results": results_df,
        "summary": summary,
        "shap_values": shap_values,
        "features": feature_cols
    }


def run_counterfactual(df, shap_results, target_group="Female_Black", baseline_group="Male_White"):

    
    if shap_results is None:
        return
    
    print("\n" + "=" * 70)
    print(f"COUNTERFACTUAL: What would {target_group} need to match {baseline_group}?")
    print("=" * 70)
    
    results_df = shap_results["results"]
    target_results = results_df[results_df["Group"] == target_group].sort_values("SHAP_Diff")
    
    target_mask = df["Group"] == target_group
    baseline_mask = df["Group"] == baseline_group
    
    for _, row in target_results.iterrows():
        if row["SHAP_Diff"] < -0.01:
            feat = row["Feature"]
            target_val = df.loc[target_mask, feat].mean()
            baseline_val = df.loc[baseline_mask, feat].mean()
            
            print(f"\n  {feat}:")
            print(f"    {target_group}: {target_val:.2f}")
            print(f"    {baseline_group}: {baseline_val:.2f}")
            print(f"    SHAP gap: {row['SHAP_Diff']:+.4f}")
            
            if target_val < baseline_val:
                pct_needed = ((baseline_val - target_val) / target_val) * 100
                print(f"    → Would need {pct_needed:.1f}% increase")
            else:
                print(f"    → NOT due to lower qualifications!")
                print(f"    → This is DISCRIMINATION in evaluation")


def generate_summary(fairness_df, odds_df, shap_results):
    
    print("\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)
    di_violations = len(fairness_df[fairness_df["DI"] < 0.8])
    eo_violations = len(fairness_df[fairness_df["EO_Gap"].abs() > 0.10])
    
    odds_groups = odds_df[odds_df["Term"].str.startswith("G_")]
    or_violations = len(odds_groups[odds_groups["Odds_Ratio"] < 0.8])
    
    print(f"\n  DI violations (< 0.8):     {di_violations}")
    print(f"  EO violations (> 10%):     {eo_violations}")
    print(f"  OR violations (< 0.8):     {or_violations}")
    print(f"  Total issues:              {di_violations + eo_violations + or_violations}")
    
    print("\n  Most disadvantaged groups:")
    worst_groups = fairness_df.nsmallest(3, "DI")
    for _, row in worst_groups.iterrows():
        print(f"    • {row['Group']}: DI = {row['DI']:.4f}, Hire Rate = {row['Hire_Rate']:.1%}")

    if shap_results:
        print("\n  Top bias-causing features (SHAP):")
        summary = shap_results["summary"]
        for feat in summary.index[:3]:
            val = summary.loc[feat, "Avg_Diff"]
            print(f"    • {feat}: {val:+.4f}")
    
    print("\n" + "-" * 50)
    if di_violations + eo_violations + or_violations > 0:
        print("  ⚠️  RECOMMENDATION: Apply bias mitigation")
        print("     Run: python c.py (Adversarial Debiasing)")
    else:
        print("  ✓ No major fairness issues detected")
    
    print("\n" + "=" * 70)
    print("Output files:")
    print("  • fairness_metrics.csv")
    print("  • odds_ratios.csv")
    print("  • shap_bias_analysis.csv")
    print("  • shap_feature_summary.csv")
    print("=" * 70)


if __name__ == "__main__":

    df = load_data("tech_diversity_hiring_data.csv")
    df = add_proxy_qualified(df)
    
    fairness_df = compute_fairness_metrics(df)
    fairness_df.to_csv("fairness_metrics.csv", index=False)
    
    odds_df = compute_odds_ratios(df)
    odds_df.to_csv("odds_ratios.csv", index=False)
    
    shap_results = run_shap_analysis(df)
    
    if shap_results:
        for group in ["Female_Black", "Female_Hispanic", "Male_Black"]:
            if group in df["Group"].values:
                run_counterfactual(df, shap_results, target_group=group)
    
    generate_summary(fairness_df, odds_df, shap_results)