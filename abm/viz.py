"""Map visualisation: a US choropleth with animated interstate migration flows.

States are shaded by net migration for the year; curved arrows show the largest
origin-to-destination flows. Alaska and Hawaii sit in the usual insets.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib import animation
from matplotlib.collections import PatchCollection
from matplotlib.patches import FancyArrowPatch, Polygon

from . import geography as geo

LAND = "#e8e6e1"
EDGE = "#ffffff"
BG = "#f7f6f4"
INK = "#2b2b2b"
ARROW = "#1f3b73"


def _state_patches(order: list[str]):
    """One matplotlib Polygon per ring, plus the index of the state it belongs to."""
    shapes = geo.load_state_shapes()
    patches, owner = [], []
    for i, abbr in enumerate(order):
        for ring in shapes.get(abbr, []):
            patches.append(Polygon(ring, closed=True))
            owner.append(i)
    return patches, np.asarray(owner)


def _diverging_norm(values: np.ndarray) -> mpl.colors.TwoSlopeNorm:
    limit = float(np.nanmax(np.abs(values)))
    limit = max(limit, 1.0)
    return mpl.colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def draw_choropleth(
    ax,
    order: list[str],
    values: np.ndarray,
    cmap: str = "RdBu_r",
    norm=None,
    label_states: bool = True,
):
    """Shade each state by `values` (aligned with `order`). Returns the collection."""
    patches, owner = _state_patches(order)
    norm = norm or _diverging_norm(values)

    collection = PatchCollection(
        patches, edgecolor=EDGE, linewidth=0.6, cmap=cmap, norm=norm, zorder=1
    )
    collection.set_array(np.asarray(values)[owner])
    ax.add_collection(collection)

    if label_states:
        cents = geo.display_centroids()
        for abbr in order:
            if abbr not in cents:
                continue
            x, y = cents[abbr]
            ax.text(
                x, y, abbr, ha="center", va="center",
                fontsize=5.5, color=INK, alpha=0.75, zorder=4,
            )

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.axis("off")
    ax.set_facecolor(BG)
    return collection


def _flow_arrows(ax, order, flows: np.ndarray, top_n: int = 25):
    """Draw the `top_n` largest net flows as curved arrows between state centroids."""
    cents = geo.display_centroids()
    # Net flow between each pair, so we draw one arrow per corridor, not two.
    net = flows - flows.T
    iu = np.triu_indices(len(order), k=1)
    magnitudes = net[iu]
    if not len(magnitudes):
        return []

    keep = np.argsort(np.abs(magnitudes))[::-1][:top_n]
    biggest = max(float(np.abs(magnitudes).max()), 1.0)

    arrows = []
    for k in keep:
        i, j = iu[0][k], iu[1][k]
        value = magnitudes[k]
        if value == 0:
            continue
        src, dst = (i, j) if value > 0 else (j, i)
        if order[src] not in cents or order[dst] not in cents:
            continue
        weight = abs(value) / biggest
        if weight < 0.08:
            continue
        arrow = FancyArrowPatch(
            cents[order[src]],
            cents[order[dst]],
            connectionstyle="arc3,rad=0.16",
            arrowstyle="-|>",
            mutation_scale=7 + 13 * weight,
            linewidth=0.5 + 3.0 * weight,
            color=ARROW,
            alpha=0.25 + 0.55 * weight,
            zorder=3,
        )
        ax.add_patch(arrow)
        arrows.append(arrow)
    return arrows


def animate_migration(
    model,
    out_path: Path | str,
    top_n: int = 25,
    fps: int = 2,
    title: str = "Young-adult interstate migration",
    subtitle: str = "",
):
    """Render the whole run to an animated file (.mp4 or .gif)."""
    import matplotlib.pyplot as plt

    order = model.order
    history = model.history
    flows = model.annual_flows()

    net = np.array([h["net_migration"] for h in history])
    norm = _diverging_norm(net[1:] if len(net) > 1 else net)

    fig, ax = plt.subplots(figsize=(12, 7.4))
    fig.patch.set_facecolor(BG)

    collection = draw_choropleth(ax, order, net[0], norm=norm)
    cbar = fig.colorbar(collection, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("Net migration of 18-34s (people/year)", fontsize=9, color=INK)
    cbar.ax.tick_params(labelsize=8, colors=INK)
    cbar.outline.set_visible(False)

    fig.suptitle(title, fontsize=15, color=INK, y=0.95)
    sub = ax.set_title(subtitle, fontsize=10, color=INK, pad=12)
    year_label = ax.text(
        0.015, 0.05, "", transform=ax.transAxes,
        fontsize=26, color=INK, alpha=0.85, fontweight="bold", va="bottom",
    )

    state = {"arrows": []}
    _, owner = _state_patches(order)

    def update(frame: int):
        collection.set_array(np.asarray(net[frame])[owner])
        for arrow in state["arrows"]:
            arrow.remove()
        state["arrows"] = _flow_arrows(ax, order, flows[frame], top_n=top_n)
        year_label.set_text(f"Year {frame}")
        return [collection, year_label, *state["arrows"]]

    anim = animation.FuncAnimation(
        fig, update, frames=len(history), interval=1000 // max(fps, 1), blit=False
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".gif":
        anim.save(out_path, writer=animation.PillowWriter(fps=fps), dpi=110)
    else:
        anim.save(out_path, writer=animation.FFMpegWriter(fps=fps, bitrate=2400), dpi=130)
    plt.close(fig)
    return out_path


def plot_snapshot(model, year: int, out_path: Path | str, top_n: int = 25):
    """A single still frame, for the report."""
    import matplotlib.pyplot as plt

    order = model.order
    snap = model.history[year]
    flows = model.annual_flows()[year]

    fig, ax = plt.subplots(figsize=(12, 7.4))
    fig.patch.set_facecolor(BG)
    collection = draw_choropleth(ax, order, snap["net_migration"])
    _flow_arrows(ax, order, flows, top_n=top_n)

    cbar = fig.colorbar(collection, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("Net migration of 18-34s (people/year)", fontsize=9, color=INK)
    cbar.outline.set_visible(False)
    ax.set_title(f"Net young-adult migration, year {year}", fontsize=14, color=INK)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out_path


def plot_validation(merged, scores: dict, out_path: Path | str):
    """Simulated vs observed interstate in-migration rate, one point per state."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    x = merged["obs_inflow_rate"]
    y = merged["sim_inflow_rate"]
    ax.scatter(x, y, s=34, color=ARROW, alpha=0.75, edgecolor="white", linewidth=0.7)

    for _, row in merged.iterrows():
        ax.annotate(
            row["abbr"], (row["obs_inflow_rate"], row["sim_inflow_rate"]),
            fontsize=6.5, alpha=0.65, xytext=(3, 3), textcoords="offset points",
        )

    lims = [0, max(float(x.max()), float(y.max())) * 1.08]
    ax.plot(lims, lims, "--", color="#999", linewidth=1, zorder=0, label="1:1")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Observed in-migration rate, ACS 18-34")
    ax.set_ylabel("Simulated in-migration rate")
    ax.set_title(
        f"Model validation   r = {scores['pearson_r']:.2f}, "
        f"rank r = {scores['spearman_r']:.2f}  (n = {scores['n_states']})",
        fontsize=12,
    )
    ax.legend(frameon=False, fontsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out_path


def plot_scenarios(measures, out_path: Path | str):
    """Key national indicators over time, averaged across seeds, one line per scenario."""
    import matplotlib.pyplot as plt

    panels = [
        ("talent_hhi", "Concentration of graduates (HHI)"),
        ("national_college_share", "National share with a degree"),
        ("mean_rent_burden", "Mean rent burden"),
        ("income_sd_across_states", "Income dispersion across states ($)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor(BG)

    scenarios = list(dict.fromkeys(measures["scenario"]))
    colours = mpl.colormaps["tab10"](np.linspace(0, 1, 10))

    for ax, (col, title) in zip(axes.ravel(), panels):
        ax.set_facecolor(BG)
        for i, name in enumerate(scenarios):
            sub = measures[measures["scenario"] == name]
            grouped = sub.groupby("year")[col]
            mean, sd = grouped.mean(), grouped.std()
            ax.plot(mean.index, mean.values, label=name, color=colours[i], linewidth=1.8)
            ax.fill_between(
                mean.index, mean - sd, mean + sd, color=colours[i], alpha=0.12, linewidth=0
            )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("year")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle(
        "Policy scenarios at equal budget (mean across seeds, band = ±1 sd)",
        fontsize=13,
    )
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out_path
