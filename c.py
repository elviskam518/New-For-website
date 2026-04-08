import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import warnings

warnings.filterwarnings("ignore")


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha=1.0):
    return GradientReversalFunction.apply(x, alpha)


def zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return s * 0.0
    return (s - s.mean()) / std


def add_proxy_qualified(
    df: pd.DataFrame,
    experience_col="YearsExperience",
    education_col="EducationLevel",
    skill_col="AlgorithmSkill",
    interview_col="OverallInterviewScore",
    score_col="QualificationScore",
    qualified_col="Qualified",
    top_quantile=0.40,
    weights=(0.15, 0.15, 0.35, 0.35),
) -> pd.DataFrame:
    df = df.copy()
    w_exp, w_edu, w_skill, w_int = weights

    score = (
        w_exp * zscore(df[experience_col])
        + w_edu * zscore(df[education_col])
        + w_skill * zscore(df[skill_col])
        + w_int * zscore(df[interview_col])
    )
    df[score_col] = score
    cutoff = df[score_col].quantile(1 - top_quantile)
    df[qualified_col] = (df[score_col] >= cutoff).astype(int)

    return df


def compute_fairness_metrics(
    df: pd.DataFrame,
    pred_col="Pred",
    group_col="Group",
    qualified_col="Qualified",
    baseline_group="Male_White",
) -> pd.DataFrame:
    results = []

    for group in df[group_col].unique():
        mask = df[group_col] == group
        group_df = df[mask]

        group_size = len(group_df)
        hire_rate = group_df[pred_col].mean()

        qualified_df = group_df[group_df[qualified_col] == 1]
        qualified_count = len(qualified_df)

        if qualified_count > 0:
            tpr_proxy = qualified_df[pred_col].mean()
        else:
            tpr_proxy = np.nan

        results.append(
            {
                "Group": group,
                "Size": group_size,
                "Hire_Rate": hire_rate,
                "Qualified_Count": qualified_count,
                "TPR_proxy": tpr_proxy,
            }
        )

    results_df = pd.DataFrame(results)

    baseline_row = results_df[results_df["Group"] == baseline_group]

    if len(baseline_row) > 0:
        baseline_hire_rate = baseline_row["Hire_Rate"].values[0]
        baseline_tpr = baseline_row["TPR_proxy"].values[0]

        results_df["DI"] = results_df["Hire_Rate"] / baseline_hire_rate
        results_df["EO_gap"] = results_df["TPR_proxy"] - baseline_tpr

    return results_df


def load_and_prepare_data(csv_path="tech_diversity_hiring_data.csv"):
    df = pd.read_csv(csv_path)

    print("=" * 70)
    print("Loaded Tech Industry Diversity Hiring Dataset")
    print("=" * 70)

    df["Group"] = df["Gender"] + "_" + df["Race"]
    df = add_proxy_qualified(df, top_quantile=0.40)

    le_group = LabelEncoder()
    df["Group_encoded"] = le_group.fit_transform(df["Group"])

    le_gender = LabelEncoder()
    df["Gender_encoded"] = le_gender.fit_transform(df["Gender"])

    feature_cols = [
        "YearsExperience",
        "EducationLevel",
        "AlgorithmSkill",
        "SystemDesignSkill",
        "OverallInterviewScore",
        "GitHubScore",
        "NumLanguages",
        "HasReferral",
        "ResumeScore",
        "TechInterviewScore",
        "CultureFitScore"
    ]

    available_cols = [c for c in feature_cols if c in df.columns]
    print(f"Features used: {available_cols}")

    X = df[available_cols].values.astype(np.float32)
    y = df["Hired"].values.astype(np.float32)
    g_intersect = df["Group_encoded"].values.astype(np.int64)
    g_gender = df["Gender_encoded"].values.astype(np.int64)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    indices = np.arange(len(df))

    # First split: train+val vs test
    (
        X_train,
        X_test,
        y_train,
        y_test,
        g_int_train,
        g_int_test,
        g_gen_train,
        g_gen_test,
        idx_train,
        idx_test,
    ) = train_test_split(
        X_scaled,
        y,
        g_intersect,
        g_gender,
        indices,
        test_size=0.2,
        random_state=42,
        stratify=g_intersect,
    )

    # Second split: train vs val
    (
        X_train,
        X_val,
        y_train,
        y_val,
        g_int_train,
        g_int_val,
        g_gen_train,
        g_gen_val,
    ) = train_test_split(
        X_train,
        y_train,
        g_int_train,
        g_gen_train,
        test_size=0.15,
        random_state=42,
        stratify=g_int_train,
    )

    X_train_t = torch.FloatTensor(X_train)
    X_val_t = torch.FloatTensor(X_val)
    X_test_t = torch.FloatTensor(X_test)

    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

    g_int_train_t = torch.LongTensor(g_int_train)
    g_int_val_t = torch.LongTensor(g_int_val)
    g_int_test_t = torch.LongTensor(g_int_test)

    g_gen_train_t = torch.LongTensor(g_gen_train)
    g_gen_val_t = torch.LongTensor(g_gen_val)
    g_gen_test_t = torch.LongTensor(g_gen_test)

    df_test = df.iloc[idx_test].copy().reset_index(drop=True)

    print("\nDataset size:")
    print(f"  - Train: {len(X_train):,}")
    print(f"  - Val:   {len(X_val):,}")
    print(f"  - Test:  {len(X_test):,}")
    print(f"  - #Features: {X_train.shape[1]}")
    print(f"  - #Intersectional groups: {len(le_group.classes_)}")
    print(f"  - Groups: {list(le_group.classes_)}")

    return {
        "X_train": X_train_t,
        "X_val": X_val_t,
        "X_test": X_test_t,
        "y_train": y_train_t,
        "y_val": y_val_t,
        "y_test": y_test_t,
        "g_int_train": g_int_train_t,
        "g_int_val": g_int_val_t,
        "g_int_test": g_int_test_t,
        "g_gen_train": g_gen_train_t,
        "g_gen_val": g_gen_val_t,
        "g_gen_test": g_gen_test_t,
        "le_group": le_group,
        "le_gender": le_gender,
        "n_groups": len(le_group.classes_),
        "n_genders": len(le_gender.classes_),
        "df_test": df_test,
        "df_full": df,
        "feature_cols": available_cols,
        "input_dim": X_train.shape[1],
        "scaler": scaler,
    }

class AdversarialDebiasingGRL(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_groups=2):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
        )

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.adversary = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_groups),
        )

    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        prediction = self.predictor(features)

        reversed_features = grad_reverse(features, alpha)
        group_logits = self.adversary(reversed_features)

        return prediction, group_logits, features


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


def train_baseline_model(model, X_train, y_train, epochs=150, batch_size=256, lr=0.001):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

def train_adversarial_model_grl(
    model,
    X_train,
    y_train,
    g_train,
    X_val,
    y_val,
    g_val,
    epochs=200,
    batch_size=256,
    lr=0.001,
    alpha_max=1.0,
    verbose=True,
):
    from sklearn.metrics import balanced_accuracy_score

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion_pred = nn.BCELoss()

    counts = torch.bincount(g_train).float()
    weights = counts.sum() / (counts + 1e-8)
    weights = weights / weights.mean()
    criterion_adv = nn.CrossEntropyLoss(weight=weights)

    train_dataset = TensorDataset(X_train, y_train, g_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    history = {
        "pred_loss": [],
        "adv_loss": [],
        "val_acc": [],
        "adv_acc": [],
        "maj_base": [],
        "adv_bal": [],
        "alpha": [],
    }

    for epoch in range(epochs):
        model.train()
        total_pred_loss = 0.0
        total_adv_loss = 0.0

        if epoch < 80:
            alpha = 0.0
        else:
            p = (epoch - 80) / (epochs - 80)
            alpha = float(alpha_max * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0))

        for X_batch, y_batch, g_batch in train_loader:
            optimizer.zero_grad()

            pred, adv_logits, _ = model(X_batch, alpha=alpha)

            pred_loss = criterion_pred(pred, y_batch)
            adv_loss = criterion_adv(adv_logits, g_batch)

            loss = pred_loss + 1.5 * adv_loss
            loss.backward()
            optimizer.step()

            total_pred_loss += pred_loss.item()
            total_adv_loss += adv_loss.item()

        avg_pred_loss = total_pred_loss / len(train_loader)
        avg_adv_loss = total_adv_loss / len(train_loader)

        do_log = verbose and (epoch < 20 or (epoch + 1) % 10 == 0)

        model.eval()
        with torch.no_grad():
            val_pred, val_adv_logits, _ = model(X_val, alpha=alpha)

            val_pred_binary = (val_pred >= 0.5).float()
            val_acc = (val_pred_binary == y_val).float().mean().item()

            adv_pred = val_adv_logits.argmax(dim=1)
            adv_acc = (adv_pred == g_val).float().mean().item()

            val_counts = torch.bincount(g_val).float()
            maj_base = (val_counts.max() / val_counts.sum()).item()

            adv_bal = balanced_accuracy_score(g_val.cpu().numpy(), adv_pred.cpu().numpy())

        history["pred_loss"].append(avg_pred_loss)
        history["adv_loss"].append(avg_adv_loss)
        history["val_acc"].append(val_acc)
        history["adv_acc"].append(adv_acc)
        history["maj_base"].append(maj_base)
        history["adv_bal"].append(adv_bal)
        history["alpha"].append(alpha)

        if do_log:
            print(
                f"  Epoch {epoch+1:3d}/{epochs} | α={alpha:.3f} | "
                f"Pred Loss: {avg_pred_loss:.4f} | "
                f"Adv Loss: {avg_adv_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Adv Acc: {adv_acc:.4f} | "
                f"MajBase: {maj_base:.4f} | "
                f"AdvBal: {adv_bal:.4f}"
            )

    return history


def evaluate_model(model, X_test, df_test, model_type="simple"):
    model.eval()
    with torch.no_grad():
        if model_type == "adversarial":
            pred, _, _ = model(X_test, alpha=0)
        else:
            pred = model(X_test)

        pred_binary = (pred >= 0.5).float().squeeze().numpy()

    df_test = df_test.copy()
    df_test["Pred"] = pred_binary.astype(int)

    results = compute_fairness_metrics(
        df_test,
        pred_col="Pred",
        group_col="Group",
        qualified_col="Qualified",
        baseline_group="Male_White",
    )

    return results, pred_binary


def run_comparison_experiment(csv_path="tech_diversity_hiring_data.csv"):
    torch.manual_seed(42)
    np.random.seed(42)
    print("\n" + "=" * 70)
    print("Tech Industry Hiring Fairness Experiment")
    print("Intersectional Adversarial Debiasing with GRL")
    print("=" * 70)

    data = load_and_prepare_data(csv_path)

    input_dim = data["X_train"].shape[1]
    n_groups = data["n_groups"]
    n_genders = data["n_genders"]

    print("\n" + "-" * 50)

    all_results = {}
    all_histories = {}

    print("\n" + "=" * 70)
    print("Experiment 1: Baseline (No Mitigation)")
    print("=" * 70)

    baseline_model = SimpleClassifier(input_dim, hidden_dim=128)
    print("\nTraining...")
    train_baseline_model(
        baseline_model,
        data["X_train"],
        data["y_train"],
        epochs=150,
        batch_size=256,
        lr=0.001,
    )

    baseline_results, baseline_pred = evaluate_model(
        baseline_model, data["X_test"], data["df_test"], model_type="simple"
    )

    baseline_acc = accuracy_score(data["y_test"].numpy(), baseline_pred)
    baseline_f1 = f1_score(data["y_test"].numpy(), baseline_pred)

    print(f"\nPerformance: Acc={baseline_acc:.4f}, F1={baseline_f1:.4f}")
    print("\nFairness metrics:")
    print(
        baseline_results[["Group", "Hire_Rate", "DI", "TPR_proxy", "EO_gap"]]
        .round(4)
        .to_string(index=False)
    )

    all_results["Baseline"] = baseline_results.copy()

    print("\n" + "=" * 70)
    print(f"Experiment 2: Adversarial Debiasing + GRL (Gender only, {n_genders} classes)")
    print("=" * 70)

    gender_adv_model = AdversarialDebiasingGRL(
        input_dim, hidden_dim=64, num_groups=n_genders
    )

    print("\nTraining...")
    gender_history = train_adversarial_model_grl(
    gender_adv_model,
    data["X_train"],
    data["y_train"],
    data["g_gen_train"],
    data["X_val"],
    data["y_val"],
    data["g_gen_val"],
    epochs=200,
    verbose=True,
)

    gender_results, gender_pred = evaluate_model(
        gender_adv_model, data["X_test"], data["df_test"], model_type="adversarial"
    )

    gender_acc = accuracy_score(data["y_test"].numpy(), gender_pred)
    gender_f1 = f1_score(data["y_test"].numpy(), gender_pred)

    print(f"\nPerformance: Acc={gender_acc:.4f}, F1={gender_f1:.4f}")
    print("\nFairness metrics:")
    print(
        gender_results[["Group", "Hire_Rate", "DI", "TPR_proxy", "EO_gap"]]
        .round(4)
        .to_string(index=False)
    )

    all_results["Gender_Adversarial"] = gender_results.copy()
    all_histories["Gender_Adversarial"] = gender_history

    print("\n" + "=" * 70)
    print(f"Experiment 3: Intersectional Adversarial Debiasing + GRL ({n_groups} groups)")
    print("=" * 70)

    intersect_adv_model = AdversarialDebiasingGRL(
        input_dim, hidden_dim=64, num_groups=n_groups
    )

    print("\nTraining...")
    intersect_history = train_adversarial_model_grl(
    intersect_adv_model,
    data["X_train"],
    data["y_train"],
    data["g_int_train"],
    data["X_val"],
    data["y_val"],
    data["g_int_val"],
    epochs=200,
    verbose=True,
)

    intersect_results, intersect_pred = evaluate_model(
        intersect_adv_model, data["X_test"], data["df_test"], model_type="adversarial"
    )

    intersect_acc = accuracy_score(data["y_test"].numpy(), intersect_pred)
    intersect_f1 = f1_score(data["y_test"].numpy(), intersect_pred)

    print(f"\nPerformance: Acc={intersect_acc:.4f}, F1={intersect_f1:.4f}")
    print("\nFairness metrics:")
    print(
        intersect_results[["Group", "Hire_Rate", "DI", "TPR_proxy", "EO_gap"]]
        .round(4)
        .to_string(index=False)
    )

    all_results["Intersectional_Adversarial"] = intersect_results.copy()
    all_histories["Intersectional_Adversarial"] = intersect_history

    print("\n" + "=" * 70)
    print("Results Comparison")
    print("=" * 70)

    comparison_di = pd.DataFrame({"Group": baseline_results["Group"]})
    comparison_di["Baseline_DI"] = baseline_results["DI"]
    comparison_di["Gender_Adv_DI"] = gender_results["DI"]
    comparison_di["Intersect_Adv_DI"] = intersect_results["DI"]

    print("\n[DI Comparison]:")
    print(comparison_di.round(4).to_string(index=False))

    comparison_hr = pd.DataFrame({"Group": baseline_results["Group"]})
    comparison_hr["Baseline_HR"] = baseline_results["Hire_Rate"]
    comparison_hr["Gender_Adv_HR"] = gender_results["Hire_Rate"]
    comparison_hr["Intersect_Adv_HR"] = intersect_results["Hire_Rate"]

    print("\n[Hiring Rate Comparison]:")
    print(comparison_hr.round(4).to_string(index=False))

    print("\n" + "=" * 70)
    print("Statistical Summary")
    print("=" * 70)

    def di_fairness(di_series):
        return (di_series - 1).abs().mean()

    baseline_di_fair = di_fairness(baseline_results["DI"])
    gender_di_fair = di_fairness(gender_results["DI"])
    intersect_di_fair = di_fairness(intersect_results["DI"])

    baseline_di_min = baseline_results["DI"].min()
    gender_di_min = gender_results["DI"].min()
    intersect_di_min = intersect_results["DI"].min()

    baseline_hr_std = baseline_results["Hire_Rate"].std()
    gender_hr_std = gender_results["Hire_Rate"].std()
    intersect_hr_std = intersect_results["Hire_Rate"].std()

    print("\n[DI Fairness] mean |DI - 1|:")
    print(f"  Baseline:              {baseline_di_fair:.4f}")
    print(f"  Gender Adversarial:    {gender_di_fair:.4f}")
    print(f"  Intersect Adversarial: {intersect_di_fair:.4f}")

    print("\n[Minimum DI]:")
    print(f"  Baseline:              {baseline_di_min:.4f}")
    print(f"  Gender Adversarial:    {gender_di_min:.4f}")
    print(f"  Intersect Adversarial: {intersect_di_min:.4f}")

    print("\n[Model Performance]:")
    print(f"  Baseline:              Acc={baseline_acc:.4f}, F1={baseline_f1:.4f}")
    print(f"  Gender Adversarial:    Acc={gender_acc:.4f}, F1={gender_f1:.4f}")
    print(f"  Intersect Adversarial: Acc={intersect_acc:.4f}, F1={intersect_f1:.4f}")

    summary = pd.DataFrame(
        {
            "Method": ["Baseline", "Gender_Adversarial", "Intersectional_Adversarial"],
            "Accuracy": [baseline_acc, gender_acc, intersect_acc],
            "F1_Score": [baseline_f1, gender_f1, intersect_f1],
            "Min_DI": [baseline_di_min, gender_di_min, intersect_di_min],
            "DI_Fairness": [baseline_di_fair, gender_di_fair, intersect_di_fair],
            "HR_Std": [baseline_hr_std, gender_hr_std, intersect_hr_std],
            "Final_Adv_Acc": [
                np.nan,
                gender_history["adv_acc"][-1],
                intersect_history["adv_acc"][-1],
            ],
        }
    )

    summary.to_csv("tech_adversarial_grl_summary.csv", index=False, encoding="utf-8-sig")
    comparison_di.to_csv(
        "tech_adversarial_grl_di_comparison.csv", index=False, encoding="utf-8-sig"
    )
    comparison_hr.to_csv(
        "tech_adversarial_grl_hr_comparison.csv", index=False, encoding="utf-8-sig"
    )

    for name, history in all_histories.items():
        hist_df = pd.DataFrame(history)
        hist_df.to_csv(
            f"tech_adversarial_grl_{name}_history.csv", index=False, encoding="utf-8-sig"
        )


    if baseline_di_fair > 0:
        di_improvement = (baseline_di_fair - intersect_di_fair) / baseline_di_fair * 100
        print(f"\n DI fairness improvement: {di_improvement:.1f}%")

    if baseline_hr_std > 0:
        hr_improvement = (baseline_hr_std - intersect_hr_std) / baseline_hr_std * 100
        print(f" Hiring-rate standard deviation improvement: {hr_improvement:.1f}%")

    acc_drop = (baseline_acc - intersect_acc) * 100
    print(f"Accuracy change: {-acc_drop:+.1f}%")

    return all_results, all_histories


if __name__ == "__main__":
    
    results, histories = run_comparison_experiment("tech_diversity_hiring_data.csv")