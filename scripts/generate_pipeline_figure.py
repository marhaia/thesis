"""Generate a pipeline architecture figure for the exposé."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis("off")
fig.patch.set_facecolor("white")

# Color palette
C_INPUT   = "#E8F4FD"
C_STAGE1  = "#D6EAF8"
C_STAGE2  = "#D5F5E3"
C_OUTPUT  = "#FEF9E7"
C_COHERE  = "#FDEDEC"
C_BORDER  = "#2C3E50"
C_ARROW   = "#5D6D7E"
C_TITLE1  = "#1A5276"
C_TITLE2  = "#1E8449"

def box(ax, x, y, w, h, fc, ec=C_BORDER, lw=1.5, radius=0.3):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0.05,rounding_size={radius}",
                          facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
    ax.add_patch(rect)

def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW,
                                lw=1.8, mutation_scale=16), zorder=4)

def text(ax, x, y, s, size=9, weight="normal", color="black", ha="center", va="center"):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color,
            ha=ha, va=va, zorder=5)

# ── INPUT ──────────────────────────────────────────────────────────────────
box(ax, 0.4, 3.8, 2.4, 1.4, C_INPUT)
text(ax, 1.6, 4.8, "INPUT", 9, "bold", C_TITLE1)
text(ax, 1.6, 4.45, "GUI Screenshot", 9)
text(ax, 1.6, 4.1, "(Automotive HMI)", 8, color="#555")

# ── STAGE 1 ────────────────────────────────────────────────────────────────
box(ax, 3.2, 2.2, 3.4, 4.6, C_STAGE1, lw=2)
text(ax, 4.9, 6.55, "STAGE 1", 10, "bold", C_TITLE1)
text(ax, 4.9, 6.2, "Task-Independent", 9, color=C_TITLE1)
text(ax, 4.9, 5.9, "Complexity Extraction", 9, color=C_TITLE1)

features = [
    "Shannon Entropy",
    "Edge Density",
    "Feature Congestion",
    "Subband Entropy",
    "Layout Symmetry",
    "Chromatic Coherence",
    "Visual Hierarchy",
    "Element Density",
]
for i, f in enumerate(features):
    y = 5.5 - i * 0.37
    box(ax, 3.35, y - 0.16, 3.1, 0.32, "white", "#AAA", lw=0.8, radius=0.1)
    text(ax, 4.9, y, f, 7.5)

text(ax, 4.9, 2.45, "v ∈ ℝ⁸", 10, "bold", C_TITLE1)

# ── TASK + PROFILE inputs ──────────────────────────────────────────────────
box(ax, 7.2, 6.5, 2.5, 0.9, C_INPUT)
text(ax, 8.45, 7.15, "Task Descriptor", 8, "bold")
text(ax, 8.45, 6.85, "Navigation / Search /", 7.5, color="#555")
text(ax, 8.45, 6.6,  "Monitoring / Data Entry /", 7.5, color="#555")
# second line
box(ax, 7.2, 5.4, 2.5, 0.8, C_INPUT)
text(ax, 8.45, 5.95, "User Profile", 8, "bold")
text(ax, 8.45, 5.65, "Big Five (optional)", 7.5, color="#555")

# ── STAGE 2 ────────────────────────────────────────────────────────────────
box(ax, 7.2, 2.2, 2.5, 3.0, C_STAGE2, lw=2)
text(ax, 8.45, 5.0, "STAGE 2", 10, "bold", C_TITLE2)
text(ax, 8.45, 4.7, "Task-Conditioned", 9, color=C_TITLE2)
text(ax, 8.45, 4.4, "Multi-Output Prediction", 9, color=C_TITLE2)
text(ax, 8.45, 4.0, "L = λ₁L_sal + λ₂L_fix", 8, color="#333")
text(ax, 8.45, 3.72, "    + λ₃L_load + λ_coh·L_coh", 8, color="#333")

box(ax, 7.35, 2.95, 2.2, 0.55, C_COHERE, "#C0392B", lw=1, radius=0.15)
text(ax, 8.45, 3.22, "Coherence Constraint", 7.5, "bold", "#C0392B")

text(ax, 8.45, 2.55, "Head 1 · Head 2 · Head 3", 7.5, color="#555")

# ── OUTPUTS ────────────────────────────────────────────────────────────────
out_data = [
    (10.2, 6.3, "Head 1", "Saliency Map",          "AUC-Judd, NSS,\nSIM, CC"),
    (10.2, 4.5, "Head 2", "Fixation Distribution", "DTW, TDE"),
    (10.2, 2.7, "Head 3", "Cognitive Load\nIndex 0–100", "Corr. NASA-TLX"),
]
for x, y, head, out, metric in out_data:
    box(ax, x, y - 0.5, 3.3, 1.35, C_OUTPUT, lw=1.5)
    text(ax, x + 1.65, y + 0.6, head, 8, "bold", "#7D6608")
    text(ax, x + 1.65, y + 0.3, out, 8.5, color="black")
    text(ax, x + 1.65, y + 0.0, metric, 7.5, color="#777")

# ── ARROWS ─────────────────────────────────────────────────────────────────
# Input → Stage 1
arrow(ax, 2.8, 4.5, 3.2, 4.5)
# Stage 1 → Stage 2
arrow(ax, 6.6, 4.5, 7.2, 4.5)
# Task Descriptor → Stage 2
arrow(ax, 8.45, 6.5, 8.45, 5.2)
# Stage 2 → Outputs
for _, y, *_ in out_data:
    arrow(ax, 9.7, 4.5, 10.2, y + 0.18)

# ── LEGEND / NOTE ──────────────────────────────────────────────────────────
text(ax, 7.0, 1.5,
     "All three output heads are jointly optimised — the coherence term couples\n"
     "saliency, fixation, and cognitive load to prevent physically inconsistent predictions.",
     7.5, color="#555", ha="left")

plt.tight_layout(pad=0.5)
out_path = "/Users/Q682780/Thesis_G/scripts/pipeline_figure.png"
plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")
