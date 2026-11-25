"""Render vector figures from the supplied analysis summaries."""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
from matplotlib.ticker import MaxNLocator


BLUE, ORANGE, TEAL = "#235788", "#ad4e18", "#187468"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 9.0, "font.weight": "bold",
    "axes.labelweight": "bold", "axes.titleweight": "bold",
    "pdf.fonttype": 42, "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False, "axes.labelpad": 5,
    "axes.titlepad": 16, "axes.linewidth": 0.85,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "savefig.facecolor": "white", "legend.frameon": False,
})


ARIAL_BOLD_FILE = os.environ.get("FIGURE_ARIAL_BOLD")


def bold_font(size):
    if ARIAL_BOLD_FILE:
        return FontProperties(fname=ARIAL_BOLD_FILE, size=size)
    return FontProperties(family=["Arial", "Liberation Sans", "DejaVu Sans"], weight="bold", size=size)


def save(fig, target, png=False, svg=False, tiff=False):
    for text in fig.findobj(match=Text):
        text.set_fontproperties(bold_font(text.get_fontsize()))
        text.set_fontweight("bold")
    fig.savefig(target, bbox_inches="tight", pad_inches=0.045)
    if png:
        fig.savefig(target.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.045)
    if svg:
        fig.savefig(target.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.045)
    if tiff:
        fig.savefig(target.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.045)
    plt.close(fig)


def panel(ax, letter):
    ax.text(0.0, 1.055, letter, transform=ax.transAxes, weight="bold", size=18,
            ha="left", va="bottom", clip_on=False, in_layout=True)


def finite_threshold(v, g, pool):
    budget = g * (2 + v)
    candidates = [
        (1 / k - 1 / g) / (1 / k - 1 / n)
        for n in range(g + 1, pool + 1) for k in range(4, n)
        if 2 * n + (1 + v) * k <= budget + 1e-9
    ]
    return min(candidates, default=1.0)


def design(target, png=False, svg=False, tiff=False):
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.2), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.06, wspace=0.035, hspace=0.02)
    ax, other = axes
    v = np.linspace(0, 8, 641)
    c = 1 + v
    ax.plot(v, (3*c + 2 + np.sqrt(8*c*(c + 1))) / (c + 2)**2,
            color="#222222", lw=1.55, ls="-", label="Continuous design")
    ax.step(v, [finite_threshold(x, 12, 13) for x in v], where="post",
            color="#555555", lw=1.35, ls="--", label="Integer design (13 eligible)")
    ax.step(v, [finite_threshold(x, 12, 27) for x in v], where="post",
            color="#888888", lw=1.35, ls=":", label="Integer design (27 eligible)")
    ax.set(xlabel="Visit-extension cost, Δv (assay units)",
           ylabel="Squared-correlation boundary (unitless)",
           xlim=(0, 8), ylim=(0.4, 1.02))
    ax.legend(loc="lower left", fontsize=8.1, handlelength=2.0, labelspacing=0.35)
    other.plot(v, 6*(3+v) + 12*(2+v), color=BLUE, lw=1.6)
    other.scatter([2, 8], [78, 186], color=BLUE, s=15, zorder=3)
    other.annotate("78", (2, 78), xytext=(7, -5), textcoords="offset points",
                   ha="left", va="top", fontsize=9)
    other.annotate("186", (8, 186), xytext=(-8, 5), textcoords="offset points",
                   ha="right", va="bottom", fontsize=9)
    other.set(xlabel="Visit-extension cost, Δv (assay units)",
              ylabel="Total assay budget (assay units)",
              xlim=(0, 8.35))
    panel(ax, "A")
    panel(other, "B")
    save(fig, target, png, svg, tiff)


def profile_group(label):
    if label.startswith(("CN-lipid", "CN-metab")):
        return "CN-lipid + CN-metab", TEAL, "o"
    if label.startswith("CN-meal"):
        return "CN-meal", BLUE, "s"
    if label.startswith("HuMet "):
        return "HuMet OGTT/OLTT", ORANGE, "^"
    raise ValueError(f"No recruitment-group marker is defined for profile: {label}")


def profiles(records, target, png=False, svg=False, tiff=False):
    fig, ax = plt.subplots(figsize=(7.3, 2.75), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.06, h_pad=0.06, wspace=0.02, hspace=0.02)
    groups = [profile_group(record["label"]) for record in records]
    for i, (record, (_, shade, marker)) in enumerate(zip(records, groups, strict=True)):
        estimate, lo, hi = map(record.get, ("estimate", "lo", "hi"))
        ax.hlines(i, lo, hi, color=shade, lw=1.25)
        ax.scatter(estimate, i, color=shade, marker=marker, s=30, zorder=3)
    ax.set(xlabel="Median feature squared correlation (unitless)", xlim=(0, 1.0))
    ax.set_yticks(range(len(records)), [x["label"] for x in records])
    ax.set_ylim(len(records) - 0.5, -0.5)
    legend_handles = [
        Line2D([], [], color=TEAL, marker="o", linestyle="None", markersize=5,
               label="CN-lipid and CN-metab"),
        Line2D([], [], color=BLUE, marker="s", linestyle="None", markersize=5,
               label="CN-meal"),
        Line2D([], [], color=ORANGE, marker="^", linestyle="None", markersize=5,
               label="HuMet challenges"),
    ]
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 1.045),
              ncol=3, fontsize=8.5, handletextpad=0.55, columnspacing=1.25)
    save(fig, target, png, svg, tiff)


def losses(sets, target, png=False, svg=False, tiff=False):
    fig = plt.figure(figsize=(7.3, 3.05))
    axes = [fig.add_axes((left, 0.18, 0.29, 0.67)) for left in (0.205, 0.690)]
    display_labels = {
        "Pilot allocation + pilot slope": "Pilot allocation\n+ pilot slope",
        "Pilot allocation + reference slope": "Pilot allocation\n+ reference slope",
        "Reference allocation + pilot slope": "Reference allocation\n+ pilot slope",
        "Reference allocation + slope": "Reference allocation\n+ reference slope",
        "Adjusted rule + shrink": "Adjusted pilot rule\n+ shrunk slope",
    }
    headings = {"Separate evaluation pool": "Held-out participants", "Full roster": "Full cohort"}
    axis_labels = {"Mask-based standardised MSE": "Standardised MSE (random draws)"}
    for idx, (ax, config) in enumerate(zip(np.atleast_1d(axes), sets, strict=True)):
        values = config["values"]
        y = np.arange(len(values))
        colors = [BLUE if v["group"] == "direct" else TEAL if v["group"] == "reference"
                  else ORANGE for v in values]
        for i, (row, shade) in enumerate(zip(values, colors, strict=True)):
            if "lo" in row:
                ax.hlines(i, row["lo"], row["hi"], color=shade, lw=1.2, zorder=1)
            ax.scatter(row["value"], i, color=shade, s=21, zorder=2)
            if "ref" in row:
                ax.annotate(row["ref"], (row["value"], i), xytext=(5, 0),
                            textcoords="offset points", va="center", size=7)
        ax.set(xlabel=axis_labels.get(config["xlabel"], config["xlabel"]),
               xlim=config.get("xlim"))
        header_x = (0.018, 0.528)[idx]
        fig.text(header_x, 0.915, chr(65 + idx), fontsize=18, weight="bold", va="center")
        fig.text(header_x + 0.044, 0.915, headings.get(config["title"], config["title"]),
                 fontsize=10, va="center")
        ax.set_yticks(y, [display_labels.get(v["label"], v["label"]) for v in values])
        ax.set_ylim(len(values) - 0.55, -0.55)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        if config.get("zero_line"):
            ax.axvline(0, color="#a8afba", lw=.8, zorder=0)
        for axis in ("x", "y"):
            ax.tick_params(axis=axis, labelsize=8.5)
        ax.grid(axis="x", alpha=0.2, zorder=0)
    save(fig, target, png, svg, tiff)


def stress(panels, target, png=False, svg=False, tiff=False):
    fig = plt.figure(figsize=(7.3, 3.35))
    grid = fig.add_gridspec(1, len(panels), left=0.09, right=0.99, bottom=0.25,
                            top=0.80, wspace=0.22)
    axes = [fig.add_subplot(grid[0, i]) for i in range(len(panels))]
    styles = {
        "Direct without pilot": ("#222222", "o", "-"),
        "Direct after pilot": ("#444444", "s", "--"),
        "Raw pilot rule": ("#666666", "^", "-."),
        "Adjusted pilot rule": ("#888888", "D", ":"),
        "Oracle": ("#333333", "v", (0, (5, 1, 1, 1))),
    }
    legend_handles = []
    for i, (ax, config) in enumerate(zip(np.atleast_1d(axes), panels, strict=True)):
        for series in config["series"]:
            label = series["label"]
            color, marker, line = styles[label]
            # A hollow symbol leaves the underlying series visible at coincident values.
            face = "none" if label in {"Oracle", "Adjusted pilot rule"} else color
            display_label = "Population reference" if label == "Oracle" else label
            plotted, = ax.plot(series["x"], series["y"], color=color, marker=marker, ms=4.0,
                               mfc=face, ls=line, lw=1.2, label=display_label)
            if i == 0:
                legend_handles.append(plotted)
        title = config["title"]
        if title.startswith("Visit cost = "):
            title = title.replace("Visit cost", "Extension cost", 1) + " assay units"
        y_label = "MSE of mean late change" if config.get("ylabel") == "MSE of late mean" else config.get("ylabel")
        ax.set(xlabel=config["xlabel"], ylabel=y_label, title=title)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.grid(axis="y", alpha=0.2)
        panel(ax, chr(65 + i))
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.515, 0.035),
               ncol=5, fontsize=8.5, frameon=False, handlelength=1.8,
               handletextpad=0.45, columnspacing=1.1)
    save(fig, target, png, svg, tiff)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--png", action="store_true", help="also save 600-dpi PNGs for Word embedding")
    parser.add_argument("--svg", action="store_true", help="also save SVGs with editable text")
    parser.add_argument("--tiff", action="store_true", help="also save 600-dpi TIFFs")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    design(args.output / "Figure_1.pdf", args.png, args.svg, args.tiff)
    if args.input:
        data = json.loads(args.input.read_text(encoding="utf-8-sig"))
        profiles(data["profiles"], args.output / "Figure_2.pdf", args.png, args.svg, args.tiff)
        losses(data["losses"], args.output / "Figure_3.pdf", args.png, args.svg, args.tiff)
        stress(data["stress"], args.output / "Figure_4.pdf", args.png, args.svg, args.tiff)
