"""Shared presentation style for Layer 2 figures."""
from __future__ import annotations

import matplotlib.pyplot as plt

NAVY = (26 / 255, 35 / 255, 78 / 255)
DARK = (16 / 255, 24 / 255, 53 / 255)
BLUE = (39 / 255, 128 / 255, 185 / 255)
RED = (231 / 255, 76 / 255, 60 / 255)
GREEN = (39 / 255, 174 / 255, 96 / 255)
ORANGE = (230 / 255, 126 / 255, 34 / 255)
PURPLE = (142 / 255, 68 / 255, 173 / 255)
GRAY = (0.55, 0.55, 0.55)
LIGHT = (0.94, 0.94, 0.94)

DS_COLOR = {
    "mitdb": BLUE,
    "mit_bih_arrhythmia": BLUE,
    "svdb": GREEN,
    "supraventricular_arrhythmia": GREEN,
    "incartdb": ORANGE,
    "incart_st_petersburg": ORANGE,
    "nstdb": PURPLE,
    "noise_stress_test": PURPLE,
}

DS_LABEL = {
    "mitdb": "MIT-BIH",
    "mit_bih_arrhythmia": "MIT-BIH",
    "svdb": "SVDB",
    "supraventricular_arrhythmia": "SVDB",
    "incartdb": "INCART",
    "incart_st_petersburg": "INCART",
    "nstdb": "NSTDB",
    "noise_stress_test": "NSTDB",
}

LABEL_COLOR = {
    "healthy": BLUE,
    "abnormal_v": RED,
    "abnormal": RED,
    "svt": ORANGE,
    "abnormal_s": ORANGE,
}

LABEL_NICE = {
    "healthy": "Healthy",
    "abnormal_v": "Ventricular",
    "abnormal": "Abnormal",
    "svt": "SVT / PAC",
    "abnormal_s": "SVT / PAC",
}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": GRAY,
        "axes.labelcolor": DARK,
        "text.color": DARK,
        "xtick.color": DARK,
        "ytick.color": DARK,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def style_axes(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(True, color=LIGHT, linewidth=0.8, alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRAY)
    ax.spines["bottom"].set_color(GRAY)
