"""Build reports/figures/hero.png: model vs disclosed validation checks (base date 2026-06-30).

Run from the repo root:  python reports/make_figures.py

Numbers:
- Disclosed diversification effect: read from model/data/disclosures_2026h1.csv (item "분산효과", KRW 100m).
  Model values: model.kics.aggregate with archetypes.CORR_MAJOR on the disclosed major risks (asserted equal).
- B, rates +100bp, available-capital change: model -0.893tn (fixed in model/tests/test_registry.py,
  test_a004_rate_response) vs disclosed -0.869tn (README validation table).
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures" / "hero.png"

# --- inline copy of the shared portfolio style (pstyle.py) ---
COLORS = {
    "primary": "#1F4E79",
    "accent": "#E07A1F",
    "muted": "#9AA5B1",
    "light": "#D6E2EE",
    "text": "#1F2933",
    "note": "#6B7785",
}
plt.rcParams.update({
    "figure.figsize": (10, 4.2),
    "figure.dpi": 160,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelcolor": COLORS["text"],
    "axes.edgecolor": "#C9D1D9",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#EDF0F3",
    "grid.linewidth": 0.8,
    "xtick.color": COLORS["text"],
    "ytick.color": COLORS["text"],
    "legend.frameon": False,
    "axes.axisbelow": True,
})

COMPANIES = [("DB손해보험", "DB"), ("KB손해보험", "KB"), ("현대해상", "Hyundai"), ("하나손해보험", "Hana")]
RATE_MODEL, RATE_DISCLOSED = -0.893, -0.869  # KRW tn, insurer B, rates +100bp


def disclosed_diversification():
    df = pd.read_csv(ROOT / "model" / "data" / "disclosures_2026h1.csv")
    d = df[df["item"] == "분산효과"].set_index("company")["value"]
    return [float(d[k]) for k, _ in COMPANIES]


RISK_ITEMS = ("요구자본-생명장기손해보험위험", "요구자본-일반손해보험위험", "요구자본-시장위험", "요구자본-신용위험")


def model_diversification():
    """Diversification effect computed by the model: correlated aggregate minus simple sum (KRW 100m)."""
    sys.path.insert(0, str(ROOT))
    from model.archetypes import CORR_MAJOR
    from model.kics import aggregate

    df = pd.read_csv(ROOT / "model" / "data" / "disclosures_2026h1.csv")
    out = []
    for company, _ in COMPANIES:
        rows = df[df["company"] == company].set_index("item")["value"]
        risks = np.array([float(rows[i]) for i in RISK_ITEMS])  # MAJOR_RISKS order
        out.append(aggregate(risks, CORR_MAJOR) - risks.sum())
    return out


def main():
    disc = disclosed_diversification()
    model = model_diversification()
    for (_, label), m, d in zip(COMPANIES, model, disc):
        assert abs(m - d) < 1.0, f"{label}: model {m:.1f} vs disclosed {d:.1f}"

    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1.7, 1]})
    fig.suptitle("Model matches disclosed diversification exactly and the +100bp rate shock within 3%", x=0.01, ha="left",
                 fontsize=13.5, fontweight="bold", color=COLORS["text"])

    # Left: diversification effect, KRW trn (disclosed in KRW 100m)
    x = range(len(COMPANIES))
    w = 0.38
    m_tn = [abs(v) / 1e4 for v in model]
    d_tn = [abs(v) / 1e4 for v in disc]
    ax1.bar([i - w / 2 for i in x], m_tn, w, color=COLORS["primary"], label="Model")
    ax1.bar([i + w / 2 for i in x], d_tn, w, color=COLORS["muted"], label="Disclosed")
    for i, v in zip(x, d_tn):
        ax1.text(i, v + 0.15, f"{v:.2f}\nidentical", ha="center", va="bottom", fontsize=8.5,
                 color=COLORS["text"])
    ax1.set_xticks(list(x), [s for _, s in COMPANIES])
    ax1.set_ylim(0, max(d_tn) * 1.35)
    ax1.set_ylabel("Diversification benefit (KRW tn)")
    ax1.set_title("Diversification effect, 4 insurers", fontsize=11.5)
    ax1.legend(loc="upper right")
    ax1.grid(axis="x", visible=False)

    # Right: +100bp rate shock, available-capital change for B
    vals = [RATE_MODEL, RATE_DISCLOSED]
    ax2.bar([0, 1], vals, 0.55, color=[COLORS["accent"], COLORS["muted"]])
    for i, v in enumerate(vals):
        ax2.text(i, v - 0.03, f"{v:.3f}", ha="center", va="top", fontsize=10, color=COLORS["text"])
    gap = (RATE_MODEL - RATE_DISCLOSED) / RATE_DISCLOSED
    ax2.text(0.5, -0.45, f"within 3%\n(gap {gap:.1%})", ha="center", va="center", fontsize=9.5,
             color=COLORS["text"])
    ax2.axhline(0, color="#C9D1D9", linewidth=0.8)
    ax2.set_xticks([0, 1], ["Model", "Disclosed"])
    ax2.set_ylim(-1.1, 0)
    ax2.set_ylabel("Change in available capital (KRW tn)")
    ax2.set_title("Insurer B, rates +100bp", fontsize=11.5)
    ax2.grid(axis="x", visible=False)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.text(0.01, -0.02,
             "Source: insurers' H1 2026 management disclosures / IR (model/data/disclosures_2026h1.csv); "
             "diversification computed by model/kics.py; rate shock from model/tests/test_registry.py. Base date 2026-06-30.",
             fontsize=8, color=COLORS["note"], ha="left", va="top")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
