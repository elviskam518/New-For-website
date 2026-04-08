"""
Latent Space Fairness Visualisation
====================================
Generates t-SNE and PCA visualisations of latent representations
to visually demonstrate the effect of fairness interventions.

Usage:
    Run after training models. Requires trained baseline and Fair CVAE models.
    Add this to the end of run_experiment() in try.py, or run standalone.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import torch
import os


def extract_latent_representations(model, data, device="cpu", model_type="cvae"):
    """
    Extract latent representations from a trained model.
    
    For CVAE: returns the mean vector mu from the encoder
    For baseline: returns the hidden layer activations
    """
    model.eval()
    model = model.to(device)
    
    X_test = data["X_test"].to(device)
    a_test = data["a_test"].to(device)
    
    with torch.no_grad():
        if model_type == "cvae":
            mu, _ = model.encode(X_test)
            z_raw = mu.cpu().numpy()
            
            # Also get projected version if available
            if hasattr(model, 'projection') and model.projection.fitted:
                z_proj = model.projection(mu).cpu().numpy()
            else:
                z_proj = z_raw.copy()
        else:
            # For baseline SimpleClassifier, extract hidden layer
            # Run through network layer by layer
            x = X_test
            for i, layer in enumerate(model.network):
                x = layer(x)
                # Stop after second BatchNorm (before second Dropout)
                if i == 6:  # After second BatchNorm+Dropout block
                    break
            z_raw = x.cpu().numpy()
            z_proj = z_raw.copy()
    
    return z_raw, z_proj


def compute_tsne(z, perplexity=30, random_state=42):
    """Compute t-SNE embedding of latent representations."""
    print(f"  Computing t-SNE (n={len(z)}, dim={z.shape[1]}, perplexity={perplexity})...")
    tsne = TSNE(
        n_components=2, 
        perplexity=perplexity, 
        random_state=random_state,
        max_iter=1000,
        learning_rate='auto',
        init='pca'
    )
    z_2d = tsne.fit_transform(z)
    return z_2d


def compute_pca(z):
    """Compute PCA embedding of latent representations."""
    pca = PCA(n_components=2, random_state=42)
    z_2d = pca.fit_transform(z)
    variance_explained = pca.explained_variance_ratio_
    print(f"  PCA variance explained: {variance_explained[0]:.3f}, {variance_explained[1]:.3f}")
    return z_2d, variance_explained


def plot_latent_by_gender(z_2d, gender_labels, title, save_path, method="t-SNE"):
    """
    Scatter plot of 2D latent space, colored by gender.
    If gender info is removed, male and female points should overlap.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    male_mask = gender_labels == 1  # Assuming 0=Female, 1=Male from LabelEncoder
    female_mask = gender_labels == 0
    
    # Check which encoding - adjust if needed
    unique_vals = np.unique(gender_labels)
    
    ax.scatter(z_2d[male_mask, 0], z_2d[male_mask, 1], 
               c='#3498db', alpha=0.3, s=8, label='Male', rasterized=True)
    ax.scatter(z_2d[female_mask, 0], z_2d[female_mask, 1], 
               c='#e74c3c', alpha=0.3, s=8, label='Female', rasterized=True)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(f'{method} Dimension 1', fontsize=11)
    ax.set_ylabel(f'{method} Dimension 2', fontsize=11)
    ax.legend(fontsize=11, markerscale=3)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_latent_by_group(z_2d, group_labels, title, save_path, method="t-SNE"):
    """
    Scatter plot of 2D latent space, colored by intersectional group.
    Shows all 8 gender x race groups.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Color map for 8 groups
    colors = {
        'Male_White': '#2196F3',
        'Male_Asian': '#4CAF50', 
        'Male_Black': '#FF9800',
        'Male_Hispanic': '#9C27B0',
        'Female_White': '#03A9F4',
        'Female_Asian': '#8BC34A',
        'Female_Black': '#FF5722',
        'Female_Hispanic': '#E91E63',
    }
    
    for group_name, color in colors.items():
        mask = group_labels == group_name
        if mask.sum() > 0:
            ax.scatter(z_2d[mask, 0], z_2d[mask, 1],
                       c=color, alpha=0.3, s=8, label=group_name, rasterized=True)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(f'{method} Dimension 1', fontsize=11)
    ax.set_ylabel(f'{method} Dimension 2', fontsize=11)
    ax.legend(fontsize=9, markerscale=3, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_comparison_gender(z_2d_before, z_2d_after, gender_labels, 
                           title_before, title_after, save_path, method="t-SNE"):
    """
    Side-by-side comparison: before vs after debiasing, colored by gender.
    This is the most important figure - shows debiasing effect visually.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    male_mask = gender_labels == 1
    female_mask = gender_labels == 0
    
    # Before
    ax1.scatter(z_2d_before[male_mask, 0], z_2d_before[male_mask, 1],
                c='#3498db', alpha=0.3, s=8, label='Male', rasterized=True)
    ax1.scatter(z_2d_before[female_mask, 0], z_2d_before[female_mask, 1],
                c='#e74c3c', alpha=0.3, s=8, label='Female', rasterized=True)
    ax1.set_title(title_before, fontsize=13, fontweight='bold')
    ax1.set_xlabel(f'{method} Dim 1', fontsize=11)
    ax1.set_ylabel(f'{method} Dim 2', fontsize=11)
    ax1.legend(fontsize=11, markerscale=3)
    ax1.grid(True, alpha=0.3)
    
    # After
    ax2.scatter(z_2d_after[male_mask, 0], z_2d_after[male_mask, 1],
                c='#3498db', alpha=0.3, s=8, label='Male', rasterized=True)
    ax2.scatter(z_2d_after[female_mask, 0], z_2d_after[female_mask, 1],
                c='#e74c3c', alpha=0.3, s=8, label='Female', rasterized=True)
    ax2.set_title(title_after, fontsize=13, fontweight='bold')
    ax2.set_xlabel(f'{method} Dim 1', fontsize=11)
    ax2.set_ylabel(f'{method} Dim 2', fontsize=11)
    ax2.legend(fontsize=11, markerscale=3)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_comparison_group(z_2d_before, z_2d_after, group_labels,
                          title_before, title_after, save_path, method="t-SNE"):
    """
    Side-by-side comparison colored by intersectional group.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    colors = {
        'Male_White': '#2196F3', 'Male_Asian': '#4CAF50',
        'Male_Black': '#FF9800', 'Male_Hispanic': '#9C27B0',
        'Female_White': '#03A9F4', 'Female_Asian': '#8BC34A',
        'Female_Black': '#FF5722', 'Female_Hispanic': '#E91E63',
    }
    
    for group_name, color in colors.items():
        mask = group_labels == group_name
        if mask.sum() > 0:
            ax1.scatter(z_2d_before[mask, 0], z_2d_before[mask, 1],
                        c=color, alpha=0.3, s=8, label=group_name, rasterized=True)
            ax2.scatter(z_2d_after[mask, 0], z_2d_after[mask, 1],
                        c=color, alpha=0.3, s=8, label=group_name, rasterized=True)
    
    ax1.set_title(title_before, fontsize=13, fontweight='bold')
    ax2.set_title(title_after, fontsize=13, fontweight='bold')
    
    for ax in [ax1, ax2]:
        ax.set_xlabel(f'{method} Dim 1', fontsize=11)
        ax.set_ylabel(f'{method} Dim 2', fontsize=11)
        ax.legend(fontsize=8, markerscale=3, loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_density_overlap(z_2d, gender_labels, title, save_path):
    """
    KDE density plot showing overlap between male and female distributions.
    Higher overlap = more fair representation.
    """
    from scipy.stats import gaussian_kde
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    male_mask = gender_labels == 1
    female_mask = gender_labels == 0
    
    for dim, ax, dim_name in [(0, ax1, "Dimension 1"), (1, ax2, "Dimension 2")]:
        male_vals = z_2d[male_mask, dim]
        female_vals = z_2d[female_mask, dim]
        
        x_range = np.linspace(
            min(male_vals.min(), female_vals.min()) - 1,
            max(male_vals.max(), female_vals.max()) + 1,
            200
        )
        
        kde_male = gaussian_kde(male_vals)
        kde_female = gaussian_kde(female_vals)
        
        ax.fill_between(x_range, kde_male(x_range), alpha=0.4, color='#3498db', label='Male')
        ax.fill_between(x_range, kde_female(x_range), alpha=0.4, color='#e74c3c', label='Female')
        ax.plot(x_range, kde_male(x_range), color='#2980b9', linewidth=1.5)
        ax.plot(x_range, kde_female(x_range), color='#c0392b', linewidth=1.5)
        
        ax.set_title(f'{title} - {dim_name}', fontsize=12)
        ax.set_xlabel('Value', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def compute_separation_metrics(z, gender_labels):
    np.random.seed(42)  # 加这行
    """
    Quantitative metrics for how separated male/female are in latent space.
    Lower = more fair.
    """
    male_mask = gender_labels == 1
    female_mask = gender_labels == 0
    
    male_mean = z[male_mask].mean(axis=0)
    female_mean = z[female_mask].mean(axis=0)
    
    # 1. Centroid distance
    centroid_dist = np.linalg.norm(male_mean - female_mean)
    
    # 2. Linear separability (probe accuracy)
    n = len(z)
    mid = n // 2
    best_acc = 0
    for C in [0.01, 0.1, 1.0, 10.0]:
        probe = LogisticRegression(max_iter=2000, C=C)
        probe.fit(z[:mid], gender_labels[:mid])
        acc = probe.score(z[mid:], gender_labels[mid:])
        best_acc = max(best_acc, acc)
    
    # 3. Mean MMD approximation
    male_z = z[male_mask]
    female_z = z[female_mask]
    n_sample = min(500, len(male_z), len(female_z))
    idx_m = np.random.choice(len(male_z), n_sample, replace=False)
    idx_f = np.random.choice(len(female_z), n_sample, replace=False)
    
    mm = np.mean(np.linalg.norm(male_z[idx_m][:, None] - male_z[idx_m][None, :], axis=2))
    ff = np.mean(np.linalg.norm(female_z[idx_f][:, None] - female_z[idx_f][None, :], axis=2))
    mf = np.mean(np.linalg.norm(male_z[idx_m][:, None] - female_z[idx_f][None, :], axis=2))
    mmd = mf - 0.5 * (mm + ff)
    
    return {
        "centroid_distance": centroid_dist,
        "probe_accuracy": best_acc,
        "mmd_approx": mmd,
    }


def run_latent_visualisation(
    baseline_model, cvae_model, data, device="cpu",
    output_dir="latent_vis", modes=None
):
    """
    Main function: generates all latent space visualisations.
    
    Call this after training both baseline and Fair CVAE models.
    
    Args:
        baseline_model: trained SimpleClassifier
        cvae_model: trained FairCVAE_v4 (full mode)
        data: data dict from load_and_prepare_data
        device: torch device
        output_dir: directory to save figures
        modes: dict of {mode_name: trained_model} for additional CVAE variants
    """
    os.makedirs(output_dir, exist_ok=True)
    
    gender_labels = data["a_test"].numpy()
    group_labels = data["df_test"]["Group"].values
    
    print("\n" + "=" * 70)
    print("LATENT SPACE VISUALISATION")
    print("=" * 70)
    
    # ---- Extract representations ----
    print("\n[1] Extracting latent representations...")
    
    # Baseline hidden representation
    z_baseline, _ = extract_latent_representations(
        baseline_model, data, device, model_type="baseline"
    )
    print(f"  Baseline Z shape: {z_baseline.shape}")
    
    # Fair CVAE raw and projected representations
    z_cvae_raw, z_cvae_proj = extract_latent_representations(
        cvae_model, data, device, model_type="cvae"
    )
    print(f"  CVAE Z (raw) shape: {z_cvae_raw.shape}")
    print(f"  CVAE Z (projected) shape: {z_cvae_proj.shape}")
    
    # ---- Compute separation metrics ----
    print("\n[2] Computing separation metrics...")
    
    metrics_baseline = compute_separation_metrics(z_baseline, gender_labels)
    metrics_cvae = compute_separation_metrics(z_cvae_raw, gender_labels)
    
    print(f"\n  {'Metric':<25} {'Baseline':>12} {'Fair CVAE':>12} {'Reduction':>12}")
    print(f"  {'-'*61}")
    cd_red = (1 - metrics_cvae['centroid_distance'] / metrics_baseline['centroid_distance']) * 100
    mmd_red = (1 - metrics_cvae['mmd_approx'] / metrics_baseline['mmd_approx']) * 100
    probe_red = (1 - metrics_cvae['probe_accuracy'] / metrics_baseline['probe_accuracy']) * 100
    print(f"  {'Centroid Distance':<25} {metrics_baseline['centroid_distance']:>12.4f} {metrics_cvae['centroid_distance']:>12.4f} {cd_red:>11.1f}%")
    print(f"  {'Probe Accuracy':<25} {metrics_baseline['probe_accuracy']:>12.4f} {metrics_cvae['probe_accuracy']:>12.4f} {probe_red:>11.1f}%")
    print(f"  {'MMD (approx)':<25} {metrics_baseline['mmd_approx']:>12.4f} {metrics_cvae['mmd_approx']:>12.4f} {mmd_red:>11.1f}%")
    
    # ---- t-SNE visualisations ----
    print("\n[3] Computing t-SNE embeddings...")
    
    tsne_baseline = compute_tsne(z_baseline)
    tsne_cvae_raw = compute_tsne(z_cvae_raw)
    tsne_cvae_proj = compute_tsne(z_cvae_proj)
    
    # ---- Generate plots ----
    print("\n[4] Generating visualisations...")
    
    # Fig 1: Side-by-side gender comparison (MOST IMPORTANT)
    plot_comparison_gender(
        tsne_baseline, tsne_cvae_proj, gender_labels,
        "Baseline (No Debiasing)", "Fair CVAE (Full Model)",
        os.path.join(output_dir, "fig_tsne_gender_comparison.png")
    )
    
    # Fig 2: Side-by-side intersectional group comparison
    plot_comparison_group(
        tsne_baseline, tsne_cvae_proj, group_labels,
        "Baseline (No Debiasing)", "Fair CVAE (Full Model)",
        os.path.join(output_dir, "fig_tsne_group_comparison.png")
    )
    
    # Fig 3: CVAE raw vs projected (shows effect of orthogonal projection)
    plot_comparison_gender(
        tsne_cvae_raw, tsne_cvae_proj, gender_labels,
        "CVAE Latent Z (Before Projection)", "CVAE Latent Z (After Projection)",
        os.path.join(output_dir, "fig_tsne_projection_effect.png")
    )
    
    # Fig 4: Individual plots
    plot_latent_by_gender(
        tsne_baseline, gender_labels,
        "Baseline Latent Space (by Gender)",
        os.path.join(output_dir, "fig_tsne_baseline_gender.png")
    )
    
    plot_latent_by_gender(
        tsne_cvae_proj, gender_labels,
        "Fair CVAE Latent Space (by Gender)",
        os.path.join(output_dir, "fig_tsne_cvae_gender.png")
    )
    
    plot_latent_by_group(
        tsne_baseline, group_labels,
        "Baseline Latent Space (by Intersectional Group)",
        os.path.join(output_dir, "fig_tsne_baseline_group.png")
    )
    
    plot_latent_by_group(
        tsne_cvae_proj, group_labels,
        "Fair CVAE Latent Space (by Intersectional Group)",
        os.path.join(output_dir, "fig_tsne_cvae_group.png")
    )
    
    # Fig 5: Density overlap plots
    plot_density_overlap(
        tsne_baseline, gender_labels,
        "Baseline",
        os.path.join(output_dir, "fig_density_baseline.png")
    )
    
    plot_density_overlap(
        tsne_cvae_proj, gender_labels,
        "Fair CVAE",
        os.path.join(output_dir, "fig_density_cvae.png")
    )
    
    # ---- PCA visualisation ----
    print("\n[5] Computing PCA embeddings...")
    
    pca_baseline, var_baseline = compute_pca(z_baseline)
    pca_cvae_proj, var_cvae = compute_pca(z_cvae_proj)
    
    plot_comparison_gender(
        pca_baseline, pca_cvae_proj, gender_labels,
        f"Baseline PCA (var: {var_baseline[0]:.1%}, {var_baseline[1]:.1%})",
        f"Fair CVAE PCA (var: {var_cvae[0]:.1%}, {var_cvae[1]:.1%})",
        os.path.join(output_dir, "fig_pca_gender_comparison.png"),
        method="PCA"
    )
    
    # ---- Additional CVAE variants (if provided) ----
    if modes:
        print("\n[6] Visualising ablation variants...")
        
        all_z = {"Baseline": tsne_baseline}
        
        for mode_name, mode_model in modes.items():
            z_mode_raw, z_mode_proj = extract_latent_representations(
                mode_model, data, device, model_type="cvae"
            )
            tsne_mode = compute_tsne(z_mode_proj)
            all_z[mode_name] = tsne_mode
            
            # Individual plot for each mode
            plot_latent_by_gender(
                tsne_mode, gender_labels,
                f"{mode_name} Latent Space",
                os.path.join(output_dir, f"fig_tsne_{mode_name}_gender.png")
            )
            
            # Separation metrics
            metrics_mode = compute_separation_metrics(z_mode_proj, gender_labels)
            print(f"  {mode_name}: Centroid Dist={metrics_mode['centroid_distance']:.4f}, "
                  f"Probe={metrics_mode['probe_accuracy']:.4f}")
        
        # Combined 2x2 comparison if we have all modes
        if len(modes) >= 2:
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            
            male_mask = gender_labels == 1
            female_mask = gender_labels == 0
            
            plot_configs = list(all_z.items())[:4]
            
            for idx, (name, z_2d) in enumerate(plot_configs):
                ax = axes[idx // 2, idx % 2]
                ax.scatter(z_2d[male_mask, 0], z_2d[male_mask, 1],
                           c='#3498db', alpha=0.3, s=6, label='Male', rasterized=True)
                ax.scatter(z_2d[female_mask, 0], z_2d[female_mask, 1],
                           c='#e74c3c', alpha=0.3, s=6, label='Female', rasterized=True)
                ax.set_title(name, fontsize=13, fontweight='bold')
                ax.legend(fontsize=10, markerscale=3)
                ax.grid(True, alpha=0.3)
            
            plt.suptitle("Latent Space Comparison Across Methods", 
                         fontsize=15, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "fig_tsne_all_methods.png"),
                        dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved: fig_tsne_all_methods.png")
    
    # ---- Summary ----
    print("\n" + "=" * 70)
    print("VISUALISATION COMPLETE")
    print(f"All figures saved to: {output_dir}/")
    print("=" * 70)
    
    return {
        "metrics_baseline": metrics_baseline,
        "metrics_cvae": metrics_cvae,
    }


# ============================================================
# HOW TO USE: Add this to the end of run_experiment() in try.py
# ============================================================
#
#   # After all modes have been trained, add:
#   print("\n" + "#"*80)
#   print("LATENT SPACE VISUALISATION")
#   print("#"*80)
#   
#   # Use the last trained full model
#   vis_results = run_latent_visualisation(
#       baseline_model=base,
#       cvae_model=cvae_model,  # the 'full' mode model
#       data=data,
#       device=device,
#       output_dir="latent_vis",
#   )