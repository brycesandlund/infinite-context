"""Length-sweep figures for the paper: RULER-13 and OOLONG-synth (June "chart" problems).

Numbers are copied from competitor_evals.md (gpt-5.4, base Qwen single-shot) and training_runs.md (fine-tune,
harness at an 8K per-agent budget). Edit DATA below and rerun:

    uv run python paper/iclr2027/figures/plot_evals.py

Writes figures/eval_ruler.pdf and figures/eval_oolong.pdf next to this file.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MultipleLocator, NullLocator

OUT = Path(__file__).resolve().parent

# doc length (K tokens) -> score. None / missing = not run.
DATA = {
    "ruler": {  # RULER-13 mean, string match, same 65 problems per length
        "gpt": {10: 0.954, 20: 0.938, 40: 0.954, 80: 0.954, 160: 0.929, 320: 0.920},
        "base": {10: 0.923, 20: 0.938, 40: 0.938},            # 64K Tinker serving window: >=80K does not fit
        "ours": {10: 0.949, 40: 0.929, 80: 0.885, 160: 0.868, 320: 0.850},  # 18w (t=0.2); 20K not run; 320K PROVISIONAL (64/65, final in [0.840, 0.855])
    },
    "oolong": {  # OOLONG-synth chart problems, mean of counting / user / temporal, 30 problems per length
        "gpt": {10: 0.718, 20: 0.666, 40: 0.600, 80: 0.539, 160: 0.556, 320: 0.479},
        "base": {10: 0.536, 20: 0.558, 40: 0.388},
        "ours": {10: 0.606, 20: 0.540, 40: 0.535, 80: 0.561, 160: 0.464, 320: 0.470},
        # OOLONG paper §2.3 random baseline, exact expectation on these same problems (N/|L| rounded for numeric;
        # "User A or User B" as a 2-way choice)
        "random": {10: 0.226, 20: 0.225, 40: 0.184, 80: 0.181, 160: 0.191, 320: 0.161},  # 18w (t=0.2); eval_results/raw/sft_general18w_C18w_*
    },
}

SERIES = {  # fixed categorical order (dataviz reference palette, validated): ours = slot 1
    "ours": dict(label="Ours (Qwen3.6-35B-A3B + SFT, harness)", color="#2a78d6", marker="o", ls="-"),
    "gpt": dict(label="GPT-5.4", color="#eb6834", marker="s", ls="--"),
    "base": dict(label="Qwen3.6-35B-A3B", color="#1baf7a", marker="^", ls=":"),
}
TITLES = {"ruler": "RULER", "oolong": "OOLONG-synth"}
XTICKS = [10, 20, 40, 80, 160, 320]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 8,
    "axes.edgecolor": "#8a8a85", "axes.linewidth": 0.6, "xtick.color": "#55554f", "ytick.color": "#55554f",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42,
})


def plot(bench: str, ylim, path: Path, legend_loc: str):
    fig, ax = plt.subplots(figsize=(2.7, 2.1))
    if "random" in DATA[bench]:
        pts = sorted(DATA[bench]["random"].items())
        ax.plot([k for k, _ in pts], [v for _, v in pts], color="#9a9a93", lw=1.1, ls=(0, (1.5, 1.5)),
                label="Random baseline", zorder=1)
    for key in ("gpt", "base", "ours"):          # ours drawn last, on top
        pts = sorted((k, v) for k, v in DATA[bench][key].items() if v is not None)
        s = SERIES[key]
        ax.plot([k for k, _ in pts], [v for _, v in pts], ls=s["ls"], color=s["color"], lw=1.6,
                marker=s["marker"], ms=4.5, mec="white", mew=0.7, label=s["label"], zorder=3 if key == "ours" else 2)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(XTICKS))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{k}K" for k in XTICKS])
    ax.set_xlim(8.5, 380)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(MultipleLocator(0.05 if ylim[1] - ylim[0] <= 0.25 else 0.1))
    ax.set_xlabel("Document length (tokens)")
    ax.set_ylabel("Score")
    ax.set_title(TITLES[bench], fontsize=9)
    ax.grid(axis="y", color="#e4e3dd", lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if legend_loc:
        ax.legend(loc=legend_loc, fontsize=6, frameon=False, handlelength=2.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


plot("ruler", (0.80, 1.0), OUT / "eval_ruler.pdf", legend_loc="lower left")
plot("oolong", (0.10, 1.00), OUT / "eval_oolong.pdf", legend_loc="upper right")
print("wrote", OUT / "eval_ruler.pdf", OUT / "eval_oolong.pdf")
