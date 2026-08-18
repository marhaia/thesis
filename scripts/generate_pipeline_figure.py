"""Generate the current Stage-1 + Stage-2 v1 technical pipeline figure.

The diagram is a point-of-use description of the active production contract.
Stage 2 v1 is a qualitative, non-score-bearing scenario proxy. It contains no
personality input, trained regressor, or numeric task/profile modifier.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUTPUT_PATH = Path(__file__).with_name("pipeline_figure.png")

C_INPUT = "#E8F4FD"
C_STAGE1 = "#D6EAF8"
C_STAGE2 = "#D5F5E3"
C_DIAGNOSTIC = "#FFF4D6"
C_OUTPUT = "#F4ECF7"
C_BORDER = "#2C3E50"
C_ARROW = "#5D6D7E"
C_TITLE1 = "#1A5276"
C_TITLE2 = "#1E8449"


def box(ax, x, y, width, height, color, linewidth=1.5):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.06,rounding_size=0.18",
        facecolor=color,
        edgecolor=C_BORDER,
        linewidth=linewidth,
        zorder=3,
    )
    ax.add_patch(patch)


def label(
    ax,
    x,
    y,
    value,
    size=8,
    weight="normal",
    color="black",
    horizontal="center",
):
    ax.text(
        x,
        y,
        value,
        fontsize=size,
        fontweight=weight,
        color=color,
        ha=horizontal,
        va="center",
        zorder=5,
    )


def arrow(ax, start, end):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": C_ARROW,
            "linewidth": 1.6,
            "mutation_scale": 14,
        },
        zorder=4,
    )


def build_figure():
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    label(
        ax,
        7.5,
        8.55,
        "Current Technical Pipeline — Stage 1 + Stage 2 v1",
        15,
        "bold",
        C_BORDER,
    )
    label(
        ax,
        7.5,
        8.15,
        "Stage 1 remains screenshot-only; Stage 2 adds bounded qualitative context",
        9,
        color="#555555",
    )

    box(ax, 0.35, 4.7, 1.9, 1.15, C_INPUT)
    label(ax, 1.3, 5.48, "INPUT", 9, "bold", C_TITLE1)
    label(ax, 1.3, 5.13, "GUI screenshot", 9)
    label(ax, 1.3, 4.84, "single image", 7.5, color="#555555")

    box(ax, 2.75, 3.25, 4.1, 3.95, C_STAGE1, linewidth=2)
    label(ax, 4.8, 6.92, "STAGE 1 — FROZEN", 11, "bold", C_TITLE1)
    label(ax, 4.8, 6.56, "Screenshot-only feature extraction", 9, color=C_TITLE1)

    stage1_blocks = (
        ("Visual complexity", "v8"),
        ("UMSI++ saliency descriptors", "s5"),
        ("HCEye-derived layout proxies", "h6"),
    )
    for index, (name, dimension) in enumerate(stage1_blocks):
        y = 5.88 - index * 0.72
        box(ax, 3.05, y - 0.25, 3.5, 0.5, "white", linewidth=0.9)
        label(ax, 3.25, y, name, 8, horizontal="left")
        label(ax, 6.3, y, dimension, 8, "bold", C_TITLE1)

    label(ax, 4.8, 3.98, "x19 = [v8 | s5 | h6]", 10, "bold", C_TITLE1)
    label(
        ax,
        4.8,
        3.57,
        "Experimental layout complexity index",
        8.2,
        color="#444444",
    )

    box(ax, 7.45, 6.0, 3.05, 1.2, C_INPUT)
    label(ax, 8.98, 6.9, "SCENARIO LABELS", 9, "bold", C_TITLE2)
    label(ax, 8.98, 6.55, "task_type: 5 categories", 8)
    label(ax, 8.98, 6.22, "time_pressure: low / medium / high", 8)

    box(ax, 7.45, 4.15, 3.05, 1.45, C_STAGE2, linewidth=2)
    label(ax, 8.98, 5.3, "STAGE 2 v1", 10, "bold", C_TITLE2)
    label(ax, 8.98, 4.95, "Deterministic scenario proxy", 8.5)
    label(ax, 8.98, 4.62, "lower / baseline / higher", 8.5, "bold", C_TITLE2)
    label(ax, 8.98, 4.31, "score_bearing=false · numeric_modifier=null", 7.2)

    box(ax, 7.45, 2.05, 3.05, 1.6, C_DIAGNOSTIC)
    label(ax, 8.98, 3.34, "SEPARATE DIAGNOSTICS", 9, "bold", "#8A5A00")
    label(ax, 8.98, 2.96, "Optional Jokinen diagnostic", 8)
    label(ax, 8.98, 2.63, "Exploratory Cross-Signal Review", 8)
    label(ax, 8.98, 2.29, "tri-state · non-score-bearing · uncalibrated", 7.2)

    box(ax, 11.25, 3.1, 3.35, 3.25, C_OUTPUT, linewidth=2)
    label(ax, 12.92, 6.05, "PUBLIC OUTPUT", 10, "bold", "#6C3483")
    outputs = (
        "Stage-1 x19 + layout index",
        "Qualitative scenario direction",
        "Optional diagnostic status/result",
        "Cross-signal tri-state review cue",
        "Reproducibility metadata",
    )
    for index, value in enumerate(outputs):
        label(ax, 11.55, 5.55 - index * 0.46, f"• {value}", 8, horizontal="left")

    arrow(ax, (2.25, 5.28), (2.75, 5.28))
    # Stage-1 output and Stage-2 metadata remain parallel public outputs; the
    # Stage-1 arrow intentionally bypasses the Stage-2 proxy box.
    arrow(ax, (6.85, 5.78), (11.25, 5.78))
    arrow(ax, (8.98, 6.0), (8.98, 5.6))
    arrow(ax, (10.5, 4.87), (11.25, 4.87))
    arrow(ax, (6.85, 4.0), (7.45, 2.85))
    arrow(ax, (10.5, 2.85), (11.25, 3.75))

    box(ax, 1.25, 0.45, 12.5, 0.95, "#FAFAFA", linewidth=1)
    label(
        ax,
        7.5,
        1.08,
        "Scientific boundary",
        8.5,
        "bold",
        C_BORDER,
    )
    label(
        ax,
        7.5,
        0.72,
        "No personality input · no trained regressor · no numeric task modifier · not a validated cognitive-load measurement",
        8,
        color="#555555",
    )

    return fig


def main():
    figure = build_figure()
    figure.savefig(
        OUTPUT_PATH,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Title": "Current Technical Pipeline — Stage 1 + Stage 2 v1",
            "Description": (
                "Stage 1 screenshot-only outputs plus the deterministic, "
                "qualitative, non-score-bearing Stage 2 v1 scenario proxy."
            ),
        },
    )
    plt.close(figure)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
