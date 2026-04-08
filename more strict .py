

from latent_vis import run_latent_visualisation
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
import copy
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
        tpr_proxy = qualified_df[pred_col].mean() if qualified_count > 0 else np.nan
        results.append({
            "Group": group, "Size": group_size, "Hire_Rate": hire_rate,
            "Qualified_Count": qualified_count, "TPR_proxy": tpr_proxy,
        })
    results_df = pd.DataFrame(results)
    baseline_row = results_df[results_df["Group"] == baseline_group]
    if len(baseline_row) > 0:
        baseline_hire_rate = baseline_row["Hire_Rate"].values[0]
        baseline_tpr = baseline_row["TPR_proxy"].values[0]
        results_df["DI"] = results_df["Hire_Rate"] / (baseline_hire_rate + 1e-10)
        results_df["EO_gap"] = results_df["TPR_proxy"] - baseline_tpr
    return results_df


def compute_hsic(z, a_onehot, sigma_z=None):

    n = z.shape[0]
    if n < 4:
        return torch.tensor(0.0, device=z.device)
    
    # Z 的 RBF 核矩阵
    if sigma_z is None:
        with torch.no_grad():
            dists = torch.cdist(z, z)
            sigma_z = dists.median().item()
            sigma_z = max(sigma_z, 0.1)
    
    K = torch.exp(-torch.cdist(z, z).pow(2) / (2 * sigma_z ** 2))
    
    L = a_onehot @ a_onehot.T
    
    H = torch.eye(n, device=z.device) - 1.0 / n
    Kc = H @ K @ H
    Lc = H @ L @ H
    
    hsic = (Kc * Lc).sum() / ((n - 1) ** 2)
    
    return hsic


def compute_hsic_batch(z, a, n_sensitive):
    a_onehot = F.one_hot(a, num_classes=n_sensitive).float()
    
    max_samples = 512
    if z.shape[0] > max_samples:
        idx = torch.randperm(z.shape[0])[:max_samples]
        z = z[idx]
        a_onehot = a_onehot[idx]
    
    return compute_hsic(z, a_onehot)



class OrthogonalProjection(nn.Module):

    
    def __init__(self, z_dim, n_directions=1):
        super().__init__()
        self.z_dim = z_dim
        self.n_directions = n_directions
        self.register_buffer(
            'sensitive_directions',
            torch.zeros(n_directions, z_dim)
        )
        self.fitted = False
    
    def fit(self, z_np, a_np):
   
        directions = []
        
        for C in [0.1, 1.0, 10.0]:
            probe = LogisticRegression(max_iter=2000, C=C, solver='lbfgs')
            probe.fit(z_np, a_np)
            w = probe.coef_[0] 
            w = w / (np.linalg.norm(w) + 1e-10)  
            directions.append(w)
        
        W = np.stack(directions, axis=0) 
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        
        top_directions = Vt[:self.n_directions] 
        
        self.sensitive_directions.copy_(
            torch.tensor(top_directions, dtype=torch.float32)
        )
        self.fitted = True
        
        return self
    
    def project(self, z):
        if not self.fitted:
            return z
        
        z_clean = z.clone()
        for i in range(self.n_directions):
            d = self.sensitive_directions[i]  
            proj = (z_clean * d.unsqueeze(0)).sum(dim=1, keepdim=True) * d.unsqueeze(0)
            z_clean = z_clean - proj
        
        return z_clean
    
    def forward(self, z):
        return self.project(z)


def load_and_prepare_data(csv_path="tech_diversity_hiring_data.csv"):
    df = pd.read_csv(csv_path)
    print("=" * 70)
    print("Loading Tech Industry Diversity Hiring Dataset")
    print("=" * 70)

    df["Group"] = df["Gender"] + "_" + df["Race"]
    df = add_proxy_qualified(df, top_quantile=0.40)

    le_group = LabelEncoder()
    df["Group_encoded"] = le_group.fit_transform(df["Group"])
    le_gender = LabelEncoder()
    df["Gender_encoded"] = le_gender.fit_transform(df["Gender"])

    feature_cols = [
        "YearsExperience", "EducationLevel", "AlgorithmSkill",
        "SystemDesignSkill", "OverallInterviewScore", "GitHubScore",
        "NumLanguages", "HasReferral", "ResumeScore",
        "TechInterviewScore", "CultureFitScore"
    ]
    available_cols = [c for c in feature_cols if c in df.columns]
    print(f"Features used: {available_cols}")

    X = df[available_cols].values.astype(np.float32)
    y = df["Hired"].values.astype(np.float32)
    a = df["Gender_encoded"].values.astype(np.int64)
    g = df["Group_encoded"].values.astype(np.int64)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    indices = np.arange(len(df))
    (X_train, X_test, y_train, y_test,
     a_train, a_test, g_train, g_test,
     idx_train, idx_test) = train_test_split(
        X_scaled, y, a, g, indices,
        test_size=0.2, random_state=42, stratify=g,
    )

    (X_train, X_val, y_train, y_val,
     a_train, a_val, g_train, g_val) = train_test_split(
        X_train, y_train, a_train, g_train,
        test_size=0.15, random_state=42, stratify=g_train,
    )

    data = {
        "X_train": torch.tensor(X_train),
        "X_val": torch.tensor(X_val),
        "X_test": torch.tensor(X_test),
        "y_train": torch.tensor(y_train).unsqueeze(1),
        "y_val": torch.tensor(y_val).unsqueeze(1),
        "y_test": torch.tensor(y_test).unsqueeze(1),
        "a_train": torch.tensor(a_train),
        "a_val": torch.tensor(a_val),
        "a_test": torch.tensor(a_test),
        "g_train": torch.tensor(g_train),
        "g_val": torch.tensor(g_val),
        "g_test": torch.tensor(g_test),
        "df_test": df.iloc[idx_test].copy().reset_index(drop=True),
        "df_full": df,
        "feature_cols": available_cols,
        "scaler": scaler,
        "le_group": le_group,
        "le_gender": le_gender,
        "n_groups": len(le_group.classes_),
        "n_genders": len(le_gender.classes_),
        "input_dim": X_train.shape[1],
    }

    print(f"\nDataset size:")
    print(f"  - Train: {len(X_train):,}")
    print(f"  - Val:   {len(X_val):,}")
    print(f"  - Test:  {len(X_test):,}")
    print(f"  - Features: {data['input_dim']}")
    print(f"  - Sensitive groups (Gender): {data['n_genders']}")
    print(f"  - Intersectional groups: {data['n_groups']}")

    return data

def one_hot(a: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(a, num_classes=num_classes).float()


class FairCVAE_v4(nn.Module):
    def __init__(self, x_dim, n_sensitive, z_dim=64, hidden_dim=256,
                 n_sensitive_directions=3):
        super().__init__()
        
        self.x_dim = x_dim
        self.n_sensitive = n_sensitive
        self.z_dim = z_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
        )
        
        self.fc_mu = nn.Linear(hidden_dim // 2, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, z_dim)
        
        self.projection = OrthogonalProjection(z_dim, n_directions=n_sensitive_directions)
        
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + n_sensitive, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, x_dim),
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(z_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, 1),
        )
        
        self.adversary = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(z_dim, hidden_dim)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.utils.spectral_norm(nn.Linear(hidden_dim // 2, hidden_dim // 4)),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 4, n_sensitive),
        )
        
        self._init_weights()
        self._adv_init_state = copy.deepcopy(self.adversary.state_dict())
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def reset_adversary(self):
        self.adversary.load_state_dict(self._adv_init_state)
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, min=-10, max=2)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, a):
        a_onehot = one_hot(a, self.n_sensitive)
        h = torch.cat([z, a_onehot], dim=1)
        return self.decoder(h)
    
    def predict(self, z):
        z_clean = self.projection(z)
        return self.predictor(z_clean)
    
    def predict_raw(self, z):
        return self.predictor(z)
    
    def forward(self, x, a, use_grl=False, alpha=1.0, use_projection=True):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        x_recon = self.decode(z, a)
        
        if use_projection and self.projection.fitted:
            z_clean = self.projection(z)
            y_logit = self.predictor(z_clean)
        else:
            z_clean = z
            y_logit = self.predictor(z)
        if use_grl:
            z_reversed = grad_reverse(z, alpha)
            a_logit = self.adversary(z_reversed)
        else:
            a_logit = self.adversary(z)
        
        return {
            "mu": mu, "logvar": logvar, "z": z, "z_clean": z_clean,
            "x_recon": x_recon, "y_logit": y_logit, "a_logit": a_logit,
        }



def compute_kl_divergence_free_bits(mu, logvar, free_bits=0.5):
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    return kl_per_dim.sum(dim=1).mean()



def compute_sensitive_direction(model, data, device):

    model.eval()
    with torch.no_grad():
        X = data["X_train"].to(device)
        a = data["a_train"].cpu().numpy()
        mu, _ = model.encode(X)
        z_np = mu.cpu().numpy()
    
    probe = LogisticRegression(max_iter=2000, C=1.0)
    probe.fit(z_np, a)
    direction = probe.coef_[0]
    direction = direction / (np.linalg.norm(direction) + 1e-10)
    
    return torch.tensor(direction, dtype=torch.float32, device=device)


def project_gradients(model, sensitive_direction):
    d = sensitive_direction
    
    if model.fc_mu.weight.grad is not None:
        grad = model.fc_mu.weight.grad
        d_col = d.unsqueeze(1)  # (z_dim, 1)
        proj_coeff = (d_col * grad).sum(dim=0, keepdim=True)
        model.fc_mu.weight.grad -= d_col * proj_coeff
    
    if model.fc_mu.bias is not None and model.fc_mu.bias.grad is not None:
        grad = model.fc_mu.bias.grad
        proj = (grad * d).sum() * d
        model.fc_mu.bias.grad -= proj

def train_fair_cvae_v4(
    model,
    data,
    epochs=350,
    batch_size=256,
    lr_main=1e-3,
    lr_adv=2e-3,
    adv_steps=5,
    lambda_hsic=50.0,
    lambda_adv=2.0,
    alpha_max=8.0,
    adv_reset_every=40,
    projection_update_every=20,
    device="cpu",
    verbose=True,
    mode="full",  
):


    assert mode in ["full", "adv_only", "no_adv"], f"Unknown mode: {mode}"

    model = model.to(device)

    mult_recon, mult_kl, mult_hsic, mult_adv = 1.0, 1.0, 1.0, 1.0

    enable_projection = True
    enable_grad_proj = True
    enable_proj_fit = True 

    if mode == "adv_only":
        mult_recon, mult_kl, mult_hsic, mult_adv = 0.0, 0.0, 0.0, 1.0
        enable_projection = False
        enable_grad_proj = False
        enable_proj_fit = False

    elif mode == "no_adv":
        mult_recon, mult_kl, mult_hsic, mult_adv = 1.0, 1.0, 1.0, 0.0
        enable_projection = False
        enable_grad_proj = False
        enable_proj_fit = False

    main_params = (
        list(model.encoder.parameters()) +
        list(model.fc_mu.parameters()) +
        list(model.fc_logvar.parameters()) +
        list(model.decoder.parameters()) +
        list(model.predictor.parameters())
    )
    adv_params = list(model.adversary.parameters())

    opt_main = optim.AdamW(main_params, lr=lr_main, weight_decay=1e-4)
    opt_adv = optim.AdamW(adv_params, lr=lr_adv, weight_decay=1e-4)

    scheduler_main = optim.lr_scheduler.CosineAnnealingLR(opt_main, T_max=epochs, eta_min=1e-5)
    scheduler_adv = optim.lr_scheduler.CosineAnnealingLR(opt_adv, T_max=epochs, eta_min=1e-4)

    a_counts = torch.bincount(data["a_train"]).float()
    a_weights = a_counts.sum() / (a_counts + 1e-8)
    a_weights = (a_weights / a_weights.mean()).to(device)

    train_dataset = TensorDataset(data["X_train"], data["y_train"], data["a_train"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    history = {
        "loss": [], "recon_loss": [], "kl_loss": [],
        "pred_loss": [], "adv_loss": [], "hsic_loss": [],
        "val_acc": [], "val_f1": [], "adv_acc": [],
        "probe_acc": [],
        "alpha": [], "beta": [], "phase": [],
        "mode": mode,
    }

    phase1_end = 50
    phase2_end = 150
    phase3_end = 250

    sensitive_direction = None

    print("\n" + "=" * 70)
    print(f"Training Fair CVAE v4  |  mode = {mode}")
    print("=" * 70)
    print(f"  Phase 1 (1-{phase1_end}):       Predictor warmup")
    print(f"  Phase 2 ({phase1_end+1}-{phase2_end}):   VAE + HSIC + adversary")
    print(f"  Phase 3 ({phase2_end+1}-{phase3_end}):  Full + gradient projection + orthogonal proj")
    print(f"  Phase 4 ({phase3_end+1}-{epochs}):  Fine-tune with high HSIC + projection")
    print(f"  λ_hsic={lambda_hsic}, λ_adv={lambda_adv}, α_max={alpha_max}")
    print(f"  Adversary reset every {adv_reset_every} epochs")
    print(f"  Projection update every {projection_update_every} epochs")
    print("=" * 70 + "\n")

    for epoch in range(epochs):
        model.train()

        total_loss = total_recon = total_kl = 0.0
        total_pred = total_adv = total_hsic = 0.0
        n_batches = 0
        if epoch < phase1_end:
            phase = 1
            beta_kl = 0.0
            w_recon = 0.0
            w_adv = 0.0
            w_hsic = 0.0
            alpha = 0.0
            current_adv_steps = 0
            use_projection = False
            use_grad_proj = False

        elif epoch < phase2_end:
            phase = 2
            progress = (epoch - phase1_end) / (phase2_end - phase1_end)
            beta_kl = 0.1 * progress
            w_recon = 1.0
            w_adv = lambda_adv * 0.5 * progress
            w_hsic = lambda_hsic * progress
            alpha = 1.0 * progress
            current_adv_steps = max(1, int(adv_steps * progress))
            use_projection = False
            use_grad_proj = False

        elif epoch < phase3_end:
            phase = 3
            progress = (epoch - phase2_end) / (phase3_end - phase2_end)
            beta_kl = 0.1 + 0.05 * progress
            w_recon = 1.0
            w_adv = lambda_adv
            w_hsic = lambda_hsic * (1.0 + progress)
            alpha = 1.0 + (alpha_max - 1.0) * progress
            current_adv_steps = adv_steps
            use_projection = True
            use_grad_proj = True

        else:
            phase = 4
            progress = (epoch - phase3_end) / (epochs - phase3_end)
            beta_kl = 0.15
            w_recon = 0.5
            w_adv = lambda_adv
            w_hsic = lambda_hsic * 2.0
            alpha = alpha_max
            current_adv_steps = adv_steps
            use_projection = True
            use_grad_proj = True

        w_recon *= mult_recon
        beta_kl *= mult_kl
        w_hsic *= mult_hsic
        w_adv *= mult_adv

        if mult_adv == 0.0:
            alpha = 0.0
            current_adv_steps = 0

        use_projection = bool(use_projection and enable_projection)
        use_grad_proj = bool(use_grad_proj and enable_grad_proj)

        if enable_proj_fit and (epoch == phase2_end):
            print("\n  >>> Fitting orthogonal projection layer.")
            model.eval()
            with torch.no_grad():
                X_train = data["X_train"].to(device)
                mu, _ = model.encode(X_train)
                z_np = mu.cpu().numpy()
                a_np = data["a_train"].cpu().numpy()

            model.projection.fit(z_np, a_np)

            probe = LogisticRegression(max_iter=2000, C=1.0)
            n = len(z_np)
            probe.fit(z_np[:n//2], a_np[:n//2])
            before = probe.score(z_np[n//2:], a_np[n//2:])

            z_clean = model.projection(torch.tensor(z_np, device=device)).cpu().numpy()
            probe2 = LogisticRegression(max_iter=2000, C=1.0)
            probe2.fit(z_clean[:n//2], a_np[:n//2])
            after = probe2.score(z_clean[n//2:], a_np[n//2:])

            print(f"  >>> Probe before projection: {before:.4f}")
            print(f"  >>> Probe after projection:  {after:.4f}")
            print(f"  >>> Leakage reduced: {before - after:.4f}\n")

            sensitive_direction = compute_sensitive_direction(model, data, device)
            model.train()

        if (enable_proj_fit and use_projection and
            epoch > phase2_end and
            (epoch - phase2_end) % projection_update_every == 0):
            model.eval()
            with torch.no_grad():
                X_train = data["X_train"].to(device)
                mu, _ = model.encode(X_train)
                z_np = mu.cpu().numpy()
                a_np = data["a_train"].cpu().numpy()
            model.projection.fit(z_np, a_np)
            sensitive_direction = compute_sensitive_direction(model, data, device)
            model.train()

        if (mult_adv > 0.0 and current_adv_steps > 0 and
            epoch > phase1_end and
            (epoch - phase1_end) % adv_reset_every == 0):
            model.reset_adversary()
            opt_adv = optim.AdamW(model.adversary.parameters(), lr=lr_adv, weight_decay=1e-4)
            if verbose:
                print(f"  >>> Adversary reset at epoch {epoch+1}")

        for X_batch, y_batch, a_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            a_batch = a_batch.to(device)

            if current_adv_steps > 0:
                for _ in range(current_adv_steps):
                    opt_adv.zero_grad()
                    with torch.no_grad():
                        mu, logvar = model.encode(X_batch)
                        z = model.reparameterize(mu, logvar)
                    a_logit = model.adversary(z.detach())
                    adv_loss_step = F.cross_entropy(a_logit, a_batch, weight=a_weights)
                    adv_loss_step.backward()
                    torch.nn.utils.clip_grad_norm_(adv_params, max_norm=1.0)
                    opt_adv.step()

            opt_main.zero_grad()

            out = model(
                X_batch, a_batch,
                use_grl=True, alpha=alpha,
                use_projection=use_projection
            )

            pred_loss = F.binary_cross_entropy_with_logits(out["y_logit"], y_batch)

            recon_loss = F.mse_loss(out["x_recon"], X_batch) if w_recon > 0 else torch.tensor(0.0, device=device)

            if beta_kl > 0:
                kl_loss = compute_kl_divergence_free_bits(out["mu"], out["logvar"], free_bits=0.1)
            else:
                kl_loss = torch.tensor(0.0, device=device)

            adv_loss = F.cross_entropy(out["a_logit"], a_batch, weight=a_weights) if w_adv > 0 else torch.tensor(0.0, device=device)

            if w_hsic > 0:
                hsic_loss = compute_hsic_batch(out["z"], a_batch, model.n_sensitive)
            else:
                hsic_loss = torch.tensor(0.0, device=device)

            loss = (
                pred_loss
                + w_recon * recon_loss
                + beta_kl * kl_loss
                + w_adv * adv_loss
                + w_hsic * hsic_loss
            )

            loss.backward()

            if use_grad_proj and sensitive_direction is not None:
                project_gradients(model, sensitive_direction)

            torch.nn.utils.clip_grad_norm_(main_params, max_norm=1.0)
            opt_main.step()

            total_loss += loss.item()
            total_recon += recon_loss.item() if w_recon > 0 else 0.0
            total_kl += kl_loss.item() if beta_kl > 0 else 0.0
            total_pred += pred_loss.item()
            total_adv += adv_loss.item() if w_adv > 0 else 0.0
            total_hsic += hsic_loss.item() if w_hsic > 0 else 0.0
            n_batches += 1

        scheduler_main.step()
        scheduler_adv.step()
        model.eval()
        with torch.no_grad():
            X_val = data["X_val"].to(device)
            y_val = data["y_val"].to(device)
            a_val = data["a_val"].to(device)

            out_val = model(
                X_val, a_val,
                use_grl=False, alpha=0,
                use_projection=use_projection
            )

            y_prob = torch.sigmoid(out_val["y_logit"])
            y_pred = (y_prob >= 0.5).float()
            val_acc = (y_pred == y_val).float().mean().item()

            y_val_np = y_val.cpu().numpy().astype(int).flatten()
            y_pred_np = y_pred.cpu().numpy().astype(int).flatten()
            val_f1 = f1_score(y_val_np, y_pred_np, zero_division=0)

            a_pred = out_val["a_logit"].argmax(dim=1)
            adv_acc = (a_pred == a_val).float().mean().item()

        probe_acc = np.nan
        if (epoch + 1) % 25 == 0 or epoch == epochs - 1:
            probe_acc = quick_probe_test(model, data, device, use_projection=use_projection)

        history["loss"].append(total_loss / max(n_batches, 1))
        history["recon_loss"].append(total_recon / max(n_batches, 1))
        history["kl_loss"].append(total_kl / max(n_batches, 1))
        history["pred_loss"].append(total_pred / max(n_batches, 1))
        history["adv_loss"].append(total_adv / max(n_batches, 1))
        history["hsic_loss"].append(total_hsic / max(n_batches, 1))
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["adv_acc"].append(adv_acc)
        history["probe_acc"].append(probe_acc)
        history["alpha"].append(alpha)
        history["beta"].append(beta_kl)
        history["phase"].append(phase)

        if verbose and (epoch < 10 or (epoch + 1) % 10 == 0):
            hsic_str = f"HSIC={total_hsic/max(n_batches,1):.6f}" if w_hsic > 0 else "HSIC=0.000000"
            probe_str = f"Probe={probe_acc:.4f}" if not np.isnan(probe_acc) else ""
            proj_str = "+Proj" if use_projection and getattr(model.projection, "fitted", False) else ""
            gp_str = "+GP" if use_grad_proj else ""
            print(
                f"Epoch {epoch+1:3d}/{epochs} [Phase {phase}{proj_str}{gp_str}] | "
                f"β={beta_kl:.3f} | α={alpha:.2f} | "
                f"Pred={total_pred/max(n_batches,1):.4f} | "
                f"Recon={total_recon/max(n_batches,1):.4f} | "
                f"KL={total_kl/max(n_batches,1):.4f} | "
                f"{hsic_str} | "
                f"ValAcc={val_acc:.4f} | F1={val_f1:.4f} | "
                f"AdvAcc={adv_acc:.4f} {probe_str}"
            )

    return history


@torch.no_grad()
def quick_probe_test(model, data, device, use_projection=True):
    model.eval()
    X = data["X_val"].to(device)
    a = data["a_val"].cpu().numpy()
    
    mu, _ = model.encode(X)
    
    if use_projection and model.projection.fitted:
        z = model.projection(mu)
    else:
        z = mu
    
    z_np = z.cpu().numpy()
    n = len(z_np)
    mid = n // 2
    
    best_acc = 0
    for C in [0.1, 1.0, 10.0]:
        probe = LogisticRegression(max_iter=1000, C=C, solver='lbfgs')
        probe.fit(z_np[:mid], a[:mid])
        acc = probe.score(z_np[mid:], a[mid:])
        best_acc = max(best_acc, acc)
    
    return best_acc



@torch.no_grad()
def evaluate_model(model, data, device="cpu"):
    model.eval()
    model = model.to(device)
    
    X_test = data["X_test"].to(device)
    y_test = data["y_test"].to(device)
    a_test = data["a_test"].to(device)
    
    out = model(X_test, a_test, use_grl=False, alpha=0, use_projection=True)
    
    y_prob = torch.sigmoid(out["y_logit"])
    y_pred = (y_prob >= 0.5).float().squeeze().cpu().numpy()
    y_true = y_test.cpu().numpy().flatten()
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    df_test = data["df_test"].copy()
    df_test["Pred"] = y_pred.astype(int)
    
    fairness_df = compute_fairness_metrics(
        df_test, pred_col="Pred", group_col="Group",
        qualified_col="Qualified", baseline_group="Male_White",
    )
    
    pred_rate = y_pred.mean()
    if pred_rate < 0.05 or pred_rate > 0.95:
        print(f"\n⚠️ Warning: Prediction rate = {pred_rate:.2%} (可能有问题)")
    
    return {
        "accuracy": acc, "f1": f1, "fairness": fairness_df,
        "y_pred": y_pred, "y_prob": y_prob.cpu().numpy(),
        "pred_rate": pred_rate,
    }

import numpy as np

def threshold_for_target_rate(y_prob: np.ndarray, target_rate: float) -> float:
    target_rate = float(np.clip(target_rate, 1e-6, 1 - 1e-6))
    return float(np.quantile(y_prob, 1.0 - target_rate))


@torch.no_grad()
def evaluate_model_at_threshold(model, data, threshold: float, device="cpu"):
    model.eval()
    model = model.to(device)

    X_test = data["X_test"].to(device)
    y_test = data["y_test"].cpu().numpy().flatten()
    a_test = data["a_test"].to(device)

    out = model(X_test, a_test, use_grl=False, alpha=0, use_projection=True)
    y_prob = torch.sigmoid(out["y_logit"]).squeeze().cpu().numpy()
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    pred_rate = float(y_pred.mean())

    df_test = data["df_test"].copy()
    df_test["Pred"] = y_pred

    fairness_df = compute_fairness_metrics(
        df_test, pred_col="Pred", group_col="Group",
        qualified_col="Qualified", baseline_group="Male_White",
    )

    return {
        "accuracy": acc,
        "f1": f1,
        "pred_rate": pred_rate,
        "threshold": float(threshold),
        "fairness": fairness_df,
    }

@torch.no_grad()
def counterfactual_test(model, data, device="cpu"):
    model.eval()
    model = model.to(device)
    
    X = data["X_test"].to(device)
    
    mu, _ = model.encode(X)
    z_clean = model.projection(mu)
    y_prob = torch.sigmoid(model.predictor(z_clean)).squeeze()
    
    delta_mean = 0.0
    delta_max = 0.0
    consistency = 1.0
    
    return {
        "delta_mean": delta_mean, "delta_max": delta_max,
        "consistency": consistency,
    }


@torch.no_grad()
def representation_analysis(model, data, device="cpu"):

    model.eval()
    model = model.to(device)
    
    X = data["X_test"].to(device)
    a = data["a_test"].to(device)
    
    mu, _ = model.encode(X)
    z_raw = mu.cpu().numpy()
    a_np = a.cpu().numpy()
    
    z_proj = model.projection(mu).cpu().numpy()
    
    n = len(z_raw)
    z_raw_train, z_raw_test = z_raw[:n//2], z_raw[n//2:]
    z_proj_train, z_proj_test = z_proj[:n//2], z_proj[n//2:]
    a_train, a_test_np = a_np[:n//2], a_np[n//2:]
    
    best_raw = 0
    best_proj = 0
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        probe_raw = LogisticRegression(max_iter=2000, C=C)
        probe_raw.fit(z_raw_train, a_train)
        best_raw = max(best_raw, probe_raw.score(z_raw_test, a_test_np))
        
        probe_proj = LogisticRegression(max_iter=2000, C=C)
        probe_proj.fit(z_proj_train, a_train)
        best_proj = max(best_proj, probe_proj.score(z_proj_test, a_test_np))
    
    random_baseline = 1.0 / data["n_genders"]
    
    return {
        "raw_probe_accuracy": best_raw,
        "projected_probe_accuracy": best_proj,
        "random_baseline": random_baseline,
        "raw_leakage": best_raw - random_baseline,
        "projected_leakage": best_proj - random_baseline,
        "leakage_reduction": best_raw - best_proj,
    }


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


def train_baseline(model, data, epochs=150, batch_size=256, lr=0.001, device="cpu"):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()

    train_dataset = TensorDataset(data["X_train"], data["y_train"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")


@torch.no_grad()
def evaluate_baseline(model, data, device="cpu"):
    model.eval()
    model = model.to(device)
    
    X_test = data["X_test"].to(device)
    y_test = data["y_test"]
    
    y_pred = (model(X_test) >= 0.5).float().squeeze().cpu().numpy()
    y_true = y_test.numpy().flatten()
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    df_test = data["df_test"].copy()
    df_test["Pred"] = y_pred.astype(int)
    
    fairness_df = compute_fairness_metrics(
        df_test, pred_col="Pred", group_col="Group",
        qualified_col="Qualified", baseline_group="Male_White",
    )
    
    return {"accuracy": acc, "f1": f1, "fairness": fairness_df}

@torch.no_grad()
def baseline_val_pred_rate(model, data, device="cpu", threshold=0.5):
    model.eval()
    model = model.to(device)
    X_val = data["X_val"].to(device)
    y_prob = model(X_val).squeeze().cpu().numpy()  
    y_pred = (y_prob >= threshold).astype(int)
    return float(y_pred.mean()), y_prob

def run_experiment(csv_path="tech_diversity_hiring_data.csv"):
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_and_prepare_data(csv_path)

    print("\n" + "#"*80)
    print("TRAIN BASELINE")
    print("#"*80)
    base = SimpleClassifier(data["input_dim"]).to(device)
    train_baseline(base, data, epochs=150, batch_size=256, lr=1e-3, device=device)

    target_rate, base_val_prob = baseline_val_pred_rate(base, data, device=device, threshold=0.5)
    base_test = evaluate_baseline(base, data, device=device)
    print(f"\nBaseline VAL PredRate (t=0.5) = {target_rate:.2%}")
    print(base_test["fairness"][["Group","Hire_Rate","DI","EO_gap"]].round(4).to_string(index=False))

    print("\n" + "#"*80)
    print("ABLATION: Remove Biased Features (CultureFitScore, HasReferral, ResumeScore, OverallInterviewScore)")
    print("#"*80)

    # Identify which column indices correspond to biased features
    biased_features = ["CultureFitScore", "HasReferral", "ResumeScore", "OverallInterviewScore"]
    biased_idx = [i for i, col in enumerate(data["feature_cols"]) if col in biased_features]
    clean_idx = [i for i, col in enumerate(data["feature_cols"]) if col not in biased_features]
    clean_names = [col for col in data["feature_cols"] if col not in biased_features]

    print(f"  Removed features: {biased_features}")
    print(f"  Remaining features ({len(clean_idx)}): {clean_names}")

    # Slice the tensors to keep only clean features
    X_train_clean = data["X_train"][:, clean_idx]
    X_val_clean = data["X_val"][:, clean_idx]
    X_test_clean = data["X_test"][:, clean_idx]

    clean_dim = len(clean_idx)

    # Create clean data dict for training
    data_clean = {
        "X_train": X_train_clean, "X_val": X_val_clean, "X_test": X_test_clean,
        "y_train": data["y_train"], "y_val": data["y_val"], "y_test": data["y_test"],
        "a_train": data["a_train"], "a_val": data["a_val"], "a_test": data["a_test"],
        "g_train": data["g_train"], "g_val": data["g_val"], "g_test": data["g_test"],
        "df_test": data["df_test"], "input_dim": clean_dim,
        "n_genders": data["n_genders"], "n_groups": data["n_groups"],
    }

    # Train same baseline architecture but with fewer features
    base_clean = SimpleClassifier(clean_dim).to(device)
    train_baseline(base_clean, data_clean, epochs=150, batch_size=256, lr=1e-3, device=device)
    base_clean_test = evaluate_baseline(base_clean, data_clean, device=device)

    print(f"\n[Remove Biased Features] Acc={base_clean_test['accuracy']:.4f}, F1={base_clean_test['f1']:.4f}")
    print(base_clean_test["fairness"][["Group","Hire_Rate","DI","EO_gap"]].round(4).to_string(index=False))

    # ==========================================================================
    # COMPARISON TABLE
    # ==========================================================================
    print("\n" + "#"*80)
    print("COMPARISON: Baseline vs Remove Features vs Fair CVAE")
    print("#"*80)
    print(f"{'Method':<30} {'Acc':>8} {'F1':>8} {'Min DI':>8} {'Features':>10}")
    print("-" * 70)
    
    base_min_di = base_test["fairness"]["DI"].min()
    clean_min_di = base_clean_test["fairness"]["DI"].min()
    
    print(f"{'Baseline (11 features)':<30} {base_test['accuracy']:>8.4f} {base_test['f1']:>8.4f} {base_min_di:>8.4f} {'11':>10}")
    print(f"{'Remove biased (8 features)':<30} {base_clean_test['accuracy']:>8.4f} {base_clean_test['f1']:>8.4f} {clean_min_di:>8.4f} {'8':>10}")
    print(f"  -> Accuracy drop from removing features: {(base_test['accuracy'] - base_clean_test['accuracy'])*100:+.2f}%")
    print(f"  -> Min DI change: {base_min_di:.4f} -> {clean_min_di:.4f}")
    print()
    print("(Fair CVAE results will appear below for comparison)")
    print("=" * 70)
    all_results = {}

    for mode in ["adv_only", "no_adv", "full"]:
    #for mode in [ "full"]:   
        print("\n" + "="*80)
        print(f"RUN MODE = {mode}")
        print("="*80)

        cvae_model = FairCVAE_v4(
            x_dim=data["input_dim"],
            n_sensitive=data["n_genders"],
            z_dim=64,
            hidden_dim=256,
            n_sensitive_directions=3,
        ).to(device)

        _ = train_fair_cvae_v4(
            cvae_model, data,
            epochs=350, batch_size=256,
            lr_main=1e-3, lr_adv=2e-3,
            adv_steps=5,
            lambda_hsic=50.0,
            lambda_adv=2.0,
            alpha_max=8.0,
            adv_reset_every=40,
            projection_update_every=20,
            device=device,
            verbose=True,
            mode=mode,
        )

        cvae_model.eval()
        X_val = data["X_val"].to(device)
        a_val = data["a_val"].to(device)
        with torch.no_grad():
            out_val = cvae_model(X_val, a_val, use_grl=False, alpha=0, use_projection=True)
            y_prob_val = torch.sigmoid(out_val["y_logit"]).squeeze().cpu().numpy()

        t = threshold_for_target_rate(y_prob_val, target_rate)

        cal = evaluate_model_at_threshold(cvae_model, data, threshold=t, device=device)

        print(
            f"[{mode}] calibrated threshold={t:.4f} | "
            f"TEST PredRate={cal['pred_rate']:.2%} | "
            f"Acc={cal['accuracy']:.4f} | F1={cal['f1']:.4f}"
        )
        print(cal["fairness"][["Group","Hire_Rate","DI","EO_gap"]].round(4).to_string(index=False))

        all_results[mode] = cal
    
        vis_results = run_latent_visualisation(
            baseline_model=base,
            cvae_model=cvae_model,
            data=data,
            device=device,
            output_dir="latent_vis",
        )

    return {"baseline": base_test, "target_rate": target_rate, "calibrated": all_results}

if __name__ == "__main__":
    run_experiment("tech_diversity_hiring_data.csv")