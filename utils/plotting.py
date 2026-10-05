"""Shared plotting styles and helpers."""

INTERACTION_COLORS = {
    "weak": "#4C72B0",
    "medium": "#DD8452",
    "strong": "#55A868",
}


def spread_labels(ax, items, x, min_gap_frac=0.045):
    # Direct-label line ends at x, nudging labels apart vertically so they never overlap.
    lo, hi = ax.get_ylim(); gap = (hi - lo) * min_gap_frac
    placed = []
    for y, text in sorted(items):
        placed.append([max(y, placed[-1][0] + gap) if placed else y, y, text])
    overflow = placed[-1][0] - (hi - gap / 2)
    if overflow > 0:
        for p in placed:
            p[0] -= overflow
    for y_text, y_line, text in placed:
        moved = abs(y_text - y_line) > gap / 4
        ax.annotate(text, (x, y_line), xytext=(x + 0.35, y_text), textcoords="data", va="center",
                    fontsize=8, color="#52514e", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color="#b5b4ae", lw=0.6) if moved else None)
