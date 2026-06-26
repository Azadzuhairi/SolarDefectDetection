"""
preprocessing_test.py
=====================
SolarScan FYP — Preprocessing Visual Comparison Tool
University of Malaya

Shows three stages side by side for any image in the dataset:
  Stage 0 : Raw (simulated grayscale — original state before preprocessing)
  Stage 1 : After CLAHE (contrast enhancement)
  Stage 2 : After INFERNO colormap (final preprocessed state)

NOTE: preprocessing.py overwrites originals, so this script simulates
the "before" state by converting the already-processed image back to
grayscale — this faithfully reconstructs the raw thermal camera output.

HOW TO RUN:
  python preprocessing_test.py                  # first image in dataset
  python preprocessing_test.py --random         # random image
  python preprocessing_test.py --image path.jpg # specific image
  python preprocessing_test.py --all            # grid of 6 images
  python preprocessing_test.py --save           # save output to PNG

REQUIREMENTS:
  pip install opencv-python numpy matplotlib
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import os
import argparse
import random

# ─── SETTINGS ─────────────────────────────────
DATASET_DIR = "dataset2/train/images"
OUTPUT_DIR  = "runs/preprocessing_test"
CLAHE_CLIP  = 2.0
CLAHE_GRID  = (8, 8)

BG       = "#07080e"
SURFACE  = "#10131e"
BORDER   = "#1a1f30"
TEXT     = "#dde3f0"
MUTED    = "#3d4a5c"
ORANGE   = "#f97316"
AMBER    = "#f59e0b"
BLUE     = "#3b82f6"


# ─── PIPELINE FUNCTIONS ───────────────────────

def simulate_raw(image_bgr):
    """
    Reconstruct the raw grayscale thermal state from the
    already-processed INFERNO image by converting to grayscale.
    This is equivalent to what the thermal camera originally output.
    """
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def apply_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    return clahe.apply(gray)


def apply_inferno(gray):
    normalized   = gray.astype(np.float32) / 255.0
    inferno      = cm.get_cmap("inferno")
    colored_rgba = inferno(normalized)
    colored_rgb  = (colored_rgba[:, :, :3] * 255).astype(np.uint8)
    return colored_rgb


def run_pipeline(image_bgr):
    """Return all three stages."""
    raw     = simulate_raw(image_bgr)
    clahe   = apply_clahe(raw)
    inferno = apply_inferno(clahe)
    return raw, clahe, inferno


# ─── SINGLE IMAGE COMPARISON ──────────────────

def compare_single(image_path, save_path=None, show=True):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Could not read: {image_path}")
        return

    raw, clahe_img, inferno_img = run_pipeline(img_bgr)

    # Stats
    raw_std    = float(np.std(raw))
    clahe_std  = float(np.std(clahe_img))
    raw_mean   = float(np.mean(raw))
    clahe_mean = float(np.mean(clahe_img))
    contrast_gain = ((clahe_std - raw_std) / raw_std * 100) if raw_std > 0 else 0

    # Histograms
    hist_raw   = cv2.calcHist([raw],       [0], None, [256], [0, 256]).flatten()
    hist_clahe = cv2.calcHist([clahe_img], [0], None, [256], [0, 256]).flatten()

    # ─── Figure layout ────────────────────────
    fig = plt.figure(figsize=(18, 10), facecolor=BG)
    fig.suptitle(
     #   f"SolarScan  |  Preprocessing Comparison\n"
        f"{os.path.basename(image_path)}",
        color=TEXT, fontsize=13, fontweight="bold", y=0.99,
        fontfamily="monospace"
    )

    gs = gridspec.GridSpec(
        2, 4, figure=fig,
        hspace=0.45, wspace=0.30,
        top=0.91, bottom=0.06,
        left=0.04, right=0.97
    )

    # ── Row 0: Images ─────────────────────────
    stages  = [raw, clahe_img, inferno_img]
    titles  = [
        "Stage 0 — Raw Grayscale\n(before preprocessing)",
        "Stage 1 — After CLAHE\n(contrast enhancement)",
        "Stage 2 — After INFERNO\n(final preprocessed)",
    ]
    cmaps_  = ["gray", "gray", None]
    borders = [MUTED, AMBER, ORANGE]

    for col in range(3):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor(SURFACE)
        if cmaps_[col]:
            ax.imshow(stages[col], cmap=cmaps_[col], vmin=0, vmax=255)
        else:
            ax.imshow(stages[col])
        ax.set_title(titles[col], color=TEXT, fontsize=9.5,
                     fontweight="bold", pad=8, fontfamily="monospace")
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(borders[col])
            spine.set_linewidth(2)

    # ── Difference map ────────────────────────
    diff      = cv2.absdiff(raw, clahe_img).astype(np.float32)
    diff_norm = (diff / diff.max() * 255).astype(np.uint8) if diff.max() > 0 else diff.astype(np.uint8)
    diff_rgb  = apply_inferno(diff_norm)

    ax_diff = fig.add_subplot(gs[0, 3])
    ax_diff.set_facecolor(SURFACE)
    ax_diff.imshow(diff_rgb)
    ax_diff.set_title("CLAHE Difference Map\n(what CLAHE changed)",
                      color=TEXT, fontsize=9.5, fontweight="bold",
                      pad=8, fontfamily="monospace")
    ax_diff.axis("off")
    for spine in ax_diff.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(BLUE)
        spine.set_linewidth(2)

    # ── Row 1 col 0: Raw histogram ─────────────
    ax_h0 = fig.add_subplot(gs[1, 0])
    ax_h0.set_facecolor(SURFACE)
    ax_h0.fill_between(range(256), hist_raw, alpha=0.4, color=MUTED)
    ax_h0.plot(range(256), hist_raw, color=MUTED, linewidth=1.2)
    ax_h0.set_title("Pixel Histogram — Raw", color=TEXT,
                    fontsize=9, pad=5, fontfamily="monospace")
    ax_h0.set_xlim(0, 255)
    ax_h0.set_facecolor(SURFACE)
    ax_h0.tick_params(colors=MUTED, labelsize=7)
    ax_h0.spines[:].set_color(BORDER)
    ax_h0.set_xlabel("Pixel value  (0 = cold,  255 = hot)",
                     color=MUTED, fontsize=7)
    ax_h0.set_ylabel("Count", color=MUTED, fontsize=7)
    ax_h0.annotate(
        f"mean={raw_mean:.0f}  std={raw_std:.1f}",
        xy=(0.97, 0.94), xycoords="axes fraction",
        ha="right", va="top", color=MUTED,
        fontsize=7, fontfamily="monospace"
    )

    # ── Row 1 col 1: CLAHE histogram ──────────
    ax_h1 = fig.add_subplot(gs[1, 1])
    ax_h1.set_facecolor(SURFACE)
    ax_h1.fill_between(range(256), hist_clahe, alpha=0.4, color=AMBER)
    ax_h1.plot(range(256), hist_clahe, color=AMBER, linewidth=1.2)
    ax_h1.set_title("Pixel Histogram — After CLAHE", color=TEXT,
                    fontsize=9, pad=5, fontfamily="monospace")
    ax_h1.set_xlim(0, 255)
    ax_h1.tick_params(colors=MUTED, labelsize=7)
    ax_h1.spines[:].set_color(BORDER)
    ax_h1.set_xlabel("Pixel value  (0 = cold,  255 = hot)",
                     color=MUTED, fontsize=7)
    ax_h1.set_ylabel("Count", color=MUTED, fontsize=7)
    ax_h1.annotate(
        f"mean={clahe_mean:.0f}  std={clahe_std:.1f}",
        xy=(0.97, 0.94), xycoords="axes fraction",
        ha="right", va="top", color=MUTED,
        fontsize=7, fontfamily="monospace"
    )

    # ── Row 1 col 2: INFERNO scale bar ────────
    ax_cb = fig.add_subplot(gs[1, 2])
    ax_cb.set_facecolor(SURFACE)
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax_cb.imshow(gradient, aspect="auto", cmap="inferno",
                 extent=[0, 255, 0, 1])
    ax_cb.set_title("INFERNO Colormap Scale", color=TEXT,
                    fontsize=9, pad=5, fontfamily="monospace")
    ax_cb.set_yticks([])
    ax_cb.tick_params(colors=MUTED, labelsize=8)
    ax_cb.spines[:].set_color(BORDER)
    ax_cb.set_xticks([0, 64, 128, 192, 255])
    ax_cb.set_xticklabels(
        ["Cold\n(black)", "Cool\n(purple)", "Warm\n(red)",
         "Hot\n(orange)", "Very Hot\n(yellow-white)"],
        color=MUTED, fontsize=7
    )

    # ── Row 1 col 3: Stats panel ───────────────
    ax_st = fig.add_subplot(gs[1, 3])
    ax_st.set_facecolor(SURFACE)
    ax_st.axis("off")
    ax_st.spines[:].set_color(BORDER)

    stats = [
        ("CLAHE clip limit",  f"{CLAHE_CLIP}"),
        ("CLAHE tile grid",   f"{CLAHE_GRID[0]} x {CLAHE_GRID[1]}"),
        ("",                  ""),
        ("Raw mean",          f"{raw_mean:.1f}"),
        ("CLAHE mean",        f"{clahe_mean:.1f}"),
        ("",                  ""),
        ("Raw std dev",       f"{raw_std:.1f}"),
        ("CLAHE std dev",     f"{clahe_std:.1f}"),
        ("Contrast gain",     f"+{contrast_gain:.1f}%"),
        ("",                  ""),
        ("Output channels",   "3  (RGB)"),
        ("Colormap",          "INFERNO"),
    ]

    ax_st.text(0.05, 0.97, "Processing Stats",
               transform=ax_st.transAxes,
               color=ORANGE, fontsize=9.5, fontweight="bold",
               va="top", fontfamily="monospace")

    for i, (k, v) in enumerate(stats):
        y = 0.84 - i * 0.072
        if not k:
            continue
        ax_st.text(0.05, y, k, transform=ax_st.transAxes,
                   color=MUTED, fontsize=8, va="top",
                   fontfamily="monospace")
        ax_st.text(0.62, y, v, transform=ax_st.transAxes,
                   color=TEXT, fontsize=8, va="top",
                   fontfamily="monospace", fontweight="bold")

    # ── Footer ────────────────────────────────
    fig.text(0.5, 0.01,
             "SolarScan  |  University of Malaya FYP  |  "
             "Note: 'Raw' is simulated by converting processed image back to grayscale",
             ha="center", color=MUTED, fontsize=7, fontfamily="monospace")

    plt.subplots_adjust(top=0.92)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, facecolor=BG, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    plt.close()
    return contrast_gain


# ─── MULTI-IMAGE GRID ─────────────────────────

def compare_grid(image_paths, save_path, n=6):
    paths = image_paths[:n]
    rows  = len(paths)

    fig, axes = plt.subplots(
        rows, 3, figsize=(13, 4.2 * rows), facecolor=BG
    )
    if rows == 1:
        axes = [axes]

    fig.suptitle(
        "SolarScan  |  Preprocessing Comparison Grid",
        color=TEXT, fontsize=13, fontweight="bold", y=1.01,
        fontfamily="monospace"
    )

    col_titles  = ["Raw (Grayscale)", "After CLAHE", "After INFERNO"]
    col_colors  = [MUTED, AMBER, ORANGE]

    for col in range(3):
        axes[0][col].set_title(
            col_titles[col], color=col_colors[col],
            fontsize=10, fontweight="bold", pad=8,
            fontfamily="monospace"
        )

    for row, path in enumerate(paths):
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue
        raw, clahe_img, inferno_img = run_pipeline(img_bgr)
        stages = [raw, clahe_img, inferno_img]
        cmaps_ = ["gray", "gray", None]

        for col in range(3):
            ax = axes[row][col]
            ax.set_facecolor(SURFACE)
            if cmaps_[col]:
                ax.imshow(stages[col], cmap=cmaps_[col], vmin=0, vmax=255)
            else:
                ax.imshow(stages[col])
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(
                    f"#{row+1}  {os.path.basename(path)[:22]}",
                    color=MUTED, fontsize=7, rotation=90,
                    va="center", fontfamily="monospace"
                )
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(col_colors[col])
                spine.set_linewidth(1.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=130, facecolor=BG, bbox_inches="tight")
    print(f"Grid saved: {save_path}")
    plt.close()


# ─── MAIN ─────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SolarScan — Preprocessing Visual Comparison"
    )
    parser.add_argument("--image",  type=str, default=None,
                        help="Path to a specific image")
    parser.add_argument("--random", action="store_true",
                        help="Pick a random image from the dataset")
    parser.add_argument("--all",    action="store_true",
                        help="Generate a comparison grid of 6 images")
    parser.add_argument("--save",   action="store_true",
                        help="Save output PNG to runs/preprocessing_test/")
    parser.add_argument("--n",      type=int, default=6,
                        help="Number of images for --all grid (default 6)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect images
    all_images = []
    if os.path.exists(DATASET_DIR):
        all_images = sorted([
            os.path.join(DATASET_DIR, f)
            for f in os.listdir(DATASET_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ])

    if not all_images:
        print(f"No images found in {DATASET_DIR}")
        print("Make sure dataset2/train/images/ exists.")
        return

    print(f"Found {len(all_images)} images in {DATASET_DIR}")

    # Grid mode
    if args.all:
        save_path = os.path.join(OUTPUT_DIR, "comparison_grid.png")
        compare_grid(all_images, save_path, n=args.n)
        return

    # Single image mode
    if args.image:
        image_path = args.image
    elif args.random:
        image_path = random.choice(all_images)
        print(f"Selected: {os.path.basename(image_path)}")
    else:
        image_path = all_images[0]
        print(f"Using first image: {os.path.basename(image_path)}")

    save_path = None
    if args.save:
        name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(OUTPUT_DIR, f"{name}_comparison.png")

    print("\nGenerating comparison...")
    gain = compare_single(image_path, save_path=save_path, show=True)
    print(f"Contrast gain from CLAHE: +{gain:.1f}%")
    print("\nClose the plot window to exit.")


if __name__ == "__main__":
    main()
