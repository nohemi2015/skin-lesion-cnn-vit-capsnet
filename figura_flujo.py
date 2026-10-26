import os, matplotlib; matplotlib.use("Agg"); os.makedirs("figuras", exist_ok=True)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams["font.family"] = "DejaVu Sans"
W, H = 12, 8.2
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
ink = "#1f2933"; edge = "#3e4c59"; mut = "#52606d"
def box(x, y, w, h, title, body, fc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=edge, lw=1.1))
    ax.text(x + w/2, y + h - 0.15, title, ha="center", va="top", fontsize=10.5, weight="bold", color=ink)
    ax.text(x + w/2, y + h - 0.48, body, ha="center", va="top", fontsize=9.2, color=ink, linespacing=1.38)
def arr(x1, y1, x2, y2, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, lw=1.1, color=edge, shrinkA=0, shrinkB=0, linestyle=ls))
def line(xs, ys): ax.plot(xs, ys, color=edge, lw=1.1, solid_capstyle="butt")
L, M, Rr = 2.05, 6.0, 9.95; bw = 3.5
box(M - 2.6, 7.1, 5.2, 0.98, "HAM10000", "10,015 images / 7,470 lesions\n1,954 malignant, 8,061 benign images")
line([M, M], [7.1, 6.85]); line([L, Rr], [6.85, 6.85])
for x in (L, M, Rr): arr(x, 6.85, x, 6.55)
box(L - bw/2, 5.3, bw, 1.25, "Untouched cohort", "Lesions not in the subset\n7,124 images / 5,740 lesions\n562 malignant, 6,562 benign", fc="#eef3f8")
box(M - bw/2, 5.3, bw, 1.25, "Analysis subset", "Random image-level sample (seed 2026)\n2,000 images / 1,730 lesions\n1,000 malignant, 1,000 benign", fc="#eef3f8")
box(Rr - bw/2, 5.3, bw, 1.25, "Not used", "Other images of subset lesions\n891 images\n392 malignant, 499 benign", fc="#f3f4f6")
line([M, M], [5.3, 5.0]); line([M, Rr], [5.0, 5.0]); arr(M, 5.0, M, 4.7); arr(Rr, 5.0, Rr, 4.7)
box(M - bw/2, 3.0, bw, 1.7, "Original split", "Lesion-grouped, split seed 2026\nTraining: 1,404 images / 1,210 lesions\n(725 malignant, 679 benign)\nValidation: 297 / 260 (146 M, 151 B)\nTest: 299 / 260 (129 M, 170 B)")
box(Rr - bw/2, 3.3, bw, 1.4, "Five repeated splits", "Lesion-grouped, split seeds 3001-3005\nTest: 298-311 images / 260 lesions\n47.4-54.0% malignant images")
arr(L, 5.3, L, 1.5); arr(M, 3.0, M, 1.5); arr(Rr, 3.3, Rr, 1.5)
box(L - bw/2, 0.3, bw, 1.2, "Cohort evaluation", "7 pipelines x 5 seeds\nmodels and thresholds\nfrom the original split")
box(M - bw/2, 0.3, bw, 1.2, "Original-split evaluation", "7 pipelines x 5 seeds\nselection on validation,\nevaluation on test")
box(Rr - bw/2, 0.3, bw, 1.2, "Repeated-split evaluation", "5 pipelines x 3 seeds per split\nselection on validation,\nevaluation on test")
arr(M - bw/2, 0.9, L + bw/2, 0.9, ls=(0, (4, 2)))
ax.text(0.15, 0.08, "M, malignant images; B, benign images.", fontsize=8, color=mut)
fig.savefig("figuras/Figure_1_flow.png", dpi=300); fig.savefig("figuras/Figure_1_flow.pdf")
