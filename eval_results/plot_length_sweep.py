"""Length-sweep comparison plot: our fine-tuned Qwen3.6-35B-A3B (recursive
decomposition, 8K per-agent budget) vs. gpt-5.4 (single-shot, full document),
OOLONG-synth OVERALL score across document length.

Run:  uv run python eval_results/plot_length_sweep.py
Outputs eval_results/length_sweep_ours_vs_gpt5.{png,pdf}.
Tweak the DATA block and STYLE knobs below.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- DATA (OVERALL = mean of counting/user/temporal, N=10/family, held-out) ---
CTX  = [10, 20, 40, 80]                 # document length, K tokens (log2 spacing)
OURS = [0.532, 0.514, 0.562, 0.429]     # Qwen3.6-35B-A3B + decomposition (8K budget)
GPT  = [0.561, 0.583, 0.338, 0.327]     # gpt-5.4 single-shot, full document

# --- STYLE knobs ---
C_OURS, C_GPT = "#0F6E56", "#D85A30"    # teal / coral
LABEL_OURS = "Fine-tuned Qwen3.6-35B-A3B (8K budget)"
LABEL_GPT  = "gpt-5.4 (1M budget)"
TITLE = "(Preliminary) Results"
YLABEL = "OOLONG-synth score (overall)"
XLABEL = "Document length (tokens)"
YLIM = (0.25, 0.66)
SHOW_VALUE_LABELS = True
SHOW_CROSSOVER = False
SHOW_RANDOM_BASELINE = False            # set True to add a dashed random-baseline line
RANDOM_BASELINE = 0.24                  # OOLONG paper ~0.22-0.27
OUT = "eval_results/length_sweep_ours_vs_gpt5"


def main():
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=300)

    ax.plot(CTX, OURS, "-o", color=C_OURS, lw=2.4, ms=8, label=LABEL_OURS, zorder=3)
    ax.plot(CTX, GPT,  "-s", color=C_GPT,  lw=2.4, ms=8, label=LABEL_GPT, zorder=3)

    if SHOW_VALUE_LABELS:
        for x, y in zip(CTX, OURS):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                        xytext=(0, -15), ha="center", fontsize=9, color=C_OURS)
        for x, y in zip(CTX, GPT):
            dx = 13 if x == 40 else 0   # nudge the 40K label right to clear the line
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                        xytext=(dx, 9), ha="center", fontsize=9, color=C_GPT)

    if SHOW_CROSSOVER:
        ax.annotate("crossover", xy=(28, 0.55), fontsize=10,
                    color="#444441", style="italic", ha="center")

    if SHOW_RANDOM_BASELINE:
        ax.axhline(RANDOM_BASELINE, ls=":", lw=1.2, color="#888780")
        ax.annotate("random baseline", xy=(10, RANDOM_BASELINE + 0.006),
                    fontsize=8.5, color="#5F5E5A")

    ax.set_xscale("log", base=2)
    ax.set_xticks(CTX)
    ax.set_xticklabels([f"{c}K" for c in CTX])
    ax.set_xlim(8.5, 94)
    ax.set_ylim(*YLIM)
    ax.set_xlabel(XLABEL, fontsize=11)
    ax.set_ylabel(YLABEL, fontsize=11)
    ax.set_title(TITLE, fontsize=11.5, pad=10)
    ax.grid(True, which="major", axis="both", ls="--", lw=0.5, alpha=0.45)
    ax.legend(frameon=False, fontsize=9.5, loc="lower left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}.png", bbox_inches="tight")
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
    print(f"saved -> {OUT}.png + .pdf")


if __name__ == "__main__":
    main()
