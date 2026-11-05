"""Generate the figures used in the README.

Each figure is rendered twice, once for a light background and once for a dark
one, so that GitHub can show the right version for the reader's theme.

Usage (from the repository root):

    pip install numpy matplotlib
    python docs/make_figures.py

Output: docs/images/*.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "docs" / "images"

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],  # x, y, z
        "accent": "#6250d6",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70"],
        "accent": "#9085e9",
    },
}

plt.rcParams.update(
    {
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    }
)


def load_csv(name):
    """Load a dataset file into a dict of column name -> numpy array."""
    d = np.genfromtxt(DATA / name, delimiter=",", names=True, autostrip=True)
    return {k.strip(): d[k] for k in d.dtype.names}


def unwrap_deg(angle_deg):
    """Remove the 360° jumps of a wrapped angle (e.g. 359° -> 0°) so the curve stays continuous."""
    return np.degrees(np.unwrap(np.radians(angle_deg)))


def style_axes(ax, t):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], length=0, pad=6)
    ax.grid(axis="y", color=t["grid"], linewidth=1, linestyle="-")
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------
# Figure 1: what the dataset looks like
# --------------------------------------------------------------------------

def fig_dataset(t, mode, recording="IMU_Data_5.csv", start=0, length=2000):
    d = load_csv(recording)
    sl = slice(start, start + length)
    n = np.arange(start, start + length)

    panels = [
        ("Gyroscope", "angular rate (°/s)", ["Gyro_x", "Gyro_y", "Gyro_z"], False),
        ("Accelerometer", "acceleration (m/s²)", ["Acc_x", "Acc_y", "Acc_z"], False),
        ("Magnetometer", "magnetic field (µT)", ["Mag_x", "Mag_y", "Mag_z"], False),
        ("Reference orientation (to compare against)", "Euler angle, unwrapped (°)",
         ["Euler_x", "Euler_y", "Euler_z"], True),
    ]

    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 8.6), sharex=True, dpi=160)
    fig.patch.set_facecolor(t["surface"])

    for ax, (title, unit, cols, is_angle) in zip(axes, panels):
        style_axes(ax, t)
        for col, color in zip(cols, t["series"]):
            y = d[col][sl]
            if is_angle:
                y = unwrap_deg(y)
            ax.plot(n, y, color=color, linewidth=1.4)
        ax.set_title(title, loc="left", color=t["ink"], fontweight="bold", pad=6)
        ax.set_ylabel(unit, color=t["ink2"])
        ax.margins(x=0)

    axes[-1].set_xlabel("sample index", color=t["ink2"])

    # Input / reference separator: the first three panels are what the filter
    # receives, the last one is what its output should match.
    handles = [Line2D([], [], color=c, linewidth=2.5) for c in t["series"]]
    fig.legend(handles, ["x axis", "y axis", "z axis"], loc="upper right",
               bbox_to_anchor=(0.985, 0.985), ncol=3, frameon=False,
               labelcolor=t["ink"], handlelength=1.6, columnspacing=1.4)
    fig.text(0.015, 0.985, f"{recording}: raw 9-axis readings and reference orientation",
             color=t["ink"], fontsize=13, fontweight="bold", va="top")
    fig.text(0.015, 0.952,
             f"Samples {start:,}–{start + length:,} of 30,000. The filter receives the "
             "first three signals and must reproduce the fourth.",
             color=t["ink2"], fontsize=10, va="top")

    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=1.4)
    fig.savefig(OUT / f"dataset_overview_{mode}.png", facecolor=t["surface"])
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 2: Earth frame, sensor frame and the reference vectors
# --------------------------------------------------------------------------

def rotation_matrix(axis, angle_deg):
    """Rodrigues' formula: rotation of angle_deg around a unit axis."""
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    a = np.radians(angle_deg)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k


def project(v):
    """Simple oblique projection of a 3D point onto the page (x towards the viewer-left)."""
    x, y, z = v
    return np.array([y - 0.55 * x, z - 0.32 * x])


def arrow(ax, origin, vec, color, lw=2.0, ls="-", label=None, label_offset=(0, 0),
          text_color=None, fontsize=11, weight="normal", ha="center"):
    p0 = project(origin)
    p1 = project(np.asarray(origin) + np.asarray(vec))
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14,
                                 color=color, linewidth=lw, linestyle=ls,
                                 shrinkA=0, shrinkB=0))
    if label:
        ax.text(p1[0] + label_offset[0], p1[1] + label_offset[1], label,
                color=text_color or color, fontsize=fontsize, ha=ha, va="center",
                fontweight=weight)


def fig_frames(t, mode):
    fig, ax = plt.subplots(figsize=(10, 4.7), dpi=160)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    ax.set_aspect("equal")
    ax.axis("off")

    # Left: Earth frame with gravity and the magnetic field.
    earth_o = np.array([0.0, -2.6, 0.0])
    L = 1.5
    for vec, name in zip(np.eye(3) * L, ["x  (north)", "y", "z  (up)"]):
        offset = {"x  (north)": (-0.35, -0.22), "y": (0.25, 0.0), "z  (up)": (0.0, 0.2)}[name]
        arrow(ax, earth_o, vec, t["muted"], lw=1.6, label=name, label_offset=offset,
              text_color=t["ink2"], fontsize=10)

    arrow(ax, earth_o + [0, 0.0, 0.0], [0, 0, -1.25], t["ink"], lw=2.2,
          label="g  gravity", label_offset=(0.0, -0.22), fontsize=10, text_color=t["ink"])
    m = np.array([1.1, 0, -0.75])  # points north and dips downward
    arrow(ax, earth_o, m, t["accent"], lw=2.2, label="m  magnetic field",
          label_offset=(-0.15, -0.05), fontsize=10, text_color=t["ink"], ha="right")

    p = project(earth_o)
    ax.text(p[0], p[1] + 2.15, "Earth frame", color=t["ink"], fontsize=12,
            fontweight="bold", ha="center")
    ax.text(p[0], p[1] + 1.87, "fixed: gravity and north are known here", color=t["ink2"],
            fontsize=9.5, ha="center")

    # Right: the sensor frame, rotated by some unknown orientation q.
    sensor_o = np.array([0.0, 2.6, 0.0])
    R = rotation_matrix([0.4, 0.5, 1.0], 38)
    for vec, color, name in zip(R.T * L, t["series"], ["x", "y", "z"]):
        unit = vec / np.linalg.norm(vec)
        off = project(unit) * 0.22
        arrow(ax, sensor_o, vec, color, lw=2.4, label=name, label_offset=tuple(off),
              text_color=t["ink"], fontsize=11, weight="bold")

    p = project(sensor_o)
    ax.text(p[0], p[1] + 2.15, "Sensor frame", color=t["ink"], fontsize=12,
            fontweight="bold", ha="center")
    ax.text(p[0], p[1] + 1.87, "moves with the device: gyro, acc, mag measured here",
            color=t["ink2"], fontsize=9.5, ha="center")

    # The quantity being estimated.
    a = project(earth_o) + [1.2, 1.15]
    b = project(sensor_o) + [-0.9, 1.15]
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16,
                                 connectionstyle="arc3,rad=-0.3", color=t["ink2"],
                                 linewidth=1.6))
    mid = (a + b) / 2
    ax.text(mid[0], mid[1] - 0.05, "orientation  q", color=t["ink"], fontsize=12,
            fontweight="bold", ha="center")
    ax.text(mid[0], mid[1] - 0.32, "what the filter estimates", color=t["ink2"],
            fontsize=9.5, ha="center")

    ax.text(0, -1.95,
            "At rest the accelerometer measures only gravity and the magnetometer only the "
            "Earth's field.\nFinding the rotation that lines those measurements up with "
            "g and m corrects the drift of the integrated gyroscope.",
            color=t["ink2"], fontsize=9.5, ha="center", va="top", linespacing=1.5)

    ax.set_xlim(-5.2, 5.2)
    ax.set_ylim(-2.75, 2.45)
    fig.tight_layout(pad=0.6)
    fig.savefig(OUT / f"frames_{mode}.png", facecolor=t["surface"])
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for mode, theme in THEMES.items():
        fig_dataset(theme, mode)
        fig_frames(theme, mode)
    print(f"Figures written to {OUT.relative_to(ROOT)}/")
