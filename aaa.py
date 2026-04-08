import numpy as np
import matplotlib.pyplot as plt

def plot_training_schedule_dual_axis(
    epochs=350,
    phase1_end=50,
    phase2_end=150,
    phase3_end=250,
    lambda_hsic=50.0,
    alpha_max=8.0,
    save_path="fig_training_schedule_dual_axis.png"
):
    beta_vals, alpha_vals, hsic_vals = [], [], []

    for epoch in range(epochs):
        if epoch < phase1_end:
            beta_kl = 0.0
            alpha = 0.0
            w_hsic = 0.0
        elif epoch < phase2_end:
            progress = (epoch - phase1_end) / (phase2_end - phase1_end)
            beta_kl = 0.1 * progress
            alpha = 1.0 * progress
            w_hsic = lambda_hsic * progress
        elif epoch < phase3_end:
            progress = (epoch - phase2_end) / (phase3_end - phase2_end)
            beta_kl = 0.1 + 0.05 * progress
            alpha = 1.0 + (alpha_max - 1.0) * progress
            w_hsic = lambda_hsic * (1.0 + progress)
        else:
            beta_kl = 0.15
            alpha = alpha_max
            w_hsic = lambda_hsic * 2.0

        beta_vals.append(beta_kl)
        alpha_vals.append(alpha)
        hsic_vals.append(w_hsic)

    x = np.arange(1, epochs + 1)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    l1 = ax1.plot(x, beta_vals, label=r'KL weight $\beta$', linewidth=2)
    l2 = ax1.plot(x, alpha_vals, label=r'GRL strength $\alpha$', linewidth=2)
    l3 = ax2.plot(x, hsic_vals, label='HSIC weight', linewidth=2)

    for p in [phase1_end, phase2_end, phase3_end]:
        ax1.axvline(p, linestyle='--', alpha=0.5)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel(r'$\beta$ / $\alpha$')
    ax2.set_ylabel('HSIC weight')
    ax1.set_title("Training Schedule of Fair CVAE")

    lines = l1 + l2 + l3
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left')

    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

plot_training_schedule_dual_axis()