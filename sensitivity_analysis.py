"""Hyperparameter sensitivity analysis for HAXE's margin ranking loss (BERT, binary task).
Sweeps loss weight lambda and ranking margin m. Produces a 2-panel figure.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("sensitivity_results.csv")
lam = df[df.swept_param == "lambda"].sort_values("value").reset_index(drop=True)
mar = df[df.swept_param == "margin"].sort_values("value").reset_index(drop=True)

# ---- console summary -------------------------------------------------------
def rel(series, base):
    return 100 * (series - base) / base

print("=== LAMBDA SWEEP (margin fixed at 0.1) ===")
print(lam[["value", "macro_f1", "auprc", "iou_f1", "gmb_bpsn", "mean_entropy"]].to_string(index=False))
b = lam.iloc[0]  # lambda = 0 -> baseline
print(f"\n lambda 0 -> 2.0:  AUPRC {b.auprc:.3f}->{lam.iloc[-1].auprc:.3f} "
      f"(+{rel(lam.iloc[-1].auprc, b.auprc):.0f}%) | "
      f"BPSN {b.gmb_bpsn:.3f}->{lam.iloc[-1].gmb_bpsn:.3f} "
      f"({rel(lam.iloc[-1].gmb_bpsn, b.gmb_bpsn):.0f}%) | "
      f"entropy {b.mean_entropy:.2f}->{lam.iloc[-1].mean_entropy:.2f}")
print(f" macro-F1 range: {lam.macro_f1.min():.4f}-{lam.macro_f1.max():.4f} "
      f"(spread {100*(lam.macro_f1.max()-lam.macro_f1.min()):.2f} pp)")

print("\n=== MARGIN SWEEP (lambda fixed at 0.3) ===")
print(mar[["value", "macro_f1", "auprc", "iou_f1", "gmb_bpsn", "mean_entropy"]].to_string(index=False))
print(f" macro-F1 range: {mar.macro_f1.min():.4f}-{mar.macro_f1.max():.4f} "
      f"(spread {100*(mar.macro_f1.max()-mar.macro_f1.min()):.2f} pp)")
print(f" entropy m=0.05 -> 0.5: {mar.iloc[0].mean_entropy:.2f} -> {mar.iloc[-1].mean_entropy:.2f}")

# ---- figure ----------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.linewidth": 0.8})
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))

C = {"auprc": "#1f6f3d", "f1": "#3b5fa0", "bpsn": "#b03a2e", "ent": "#7d7d7d"}


def panel(ax, data, xlabel, chosen):
    x = np.arange(len(data))
    ax.plot(x, data.auprc, "o-", color=C["auprc"], label="AUPRC (explainability)", lw=1.6, ms=4)
    ax.plot(x, data.macro_f1, "s-", color=C["f1"], label="Macro-F1 (performance)", lw=1.6, ms=4)
    ax.plot(x, data.gmb_bpsn, "^-", color=C["bpsn"], label="GMB-BPSN (fairness)", lw=1.6, ms=4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in data.value])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Score")
    ax.set_ylim(0.40, 0.95)
    ax.grid(alpha=0.25, lw=0.6)

    ax2 = ax.twinx()
    ax2.plot(x, data.mean_entropy, "D--", color=C["ent"], label="Mean entropy", lw=1.4, ms=3.5)
    ax2.set_ylabel("Mean attention entropy")
    ax2.set_ylim(1.2, 3.2)

    ci = list(data.value).index(chosen)
    ax.axvline(ci, color="black", ls=":", lw=1.0, alpha=0.6)
    ax.annotate("paper config", xy=(ci, 0.915), ha="center", fontsize=7.5, style="italic")
    return ax2


a2 = panel(axes[0], lam, r"Loss weight  $\lambda$   (margin $m=0.1$)", 0.3)
panel(axes[1], mar, r"Ranking margin  $m$   (loss weight $\lambda=0.3$)", 0.1)
axes[0].set_title("(a) Sensitivity to loss weight $\\lambda$", fontsize=9.5)
axes[1].set_title("(b) Sensitivity to ranking margin $m$", fontsize=9.5)

h1, l1 = axes[0].get_legend_handles_labels()
h2, l2 = a2.get_legend_handles_labels()
fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, -0.04), fontsize=8)

fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig("sensitivity_analysis.png", dpi=300, bbox_inches="tight")
print("\nSaved sensitivity_analysis.png")
