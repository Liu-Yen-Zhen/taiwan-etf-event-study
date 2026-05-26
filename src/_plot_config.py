"""
src/_plot_config.py
-------------------
Shared matplotlib configuration for Chinese character rendering.

Usage (add to the top of every src module that calls matplotlib):

    from _plot_config import apply_chinese_style
    apply_chinese_style()

Call once at import time; the settings are global and persist for the
Python process lifetime.
"""

import matplotlib
import matplotlib.pyplot as plt


def apply_chinese_style() -> None:
    """Configure matplotlib to render Traditional Chinese with Heiti TC.

    Also disables the broken-minus workaround so "−" renders correctly.
    """
    matplotlib.rcParams.update({
        "font.family":        "Heiti TC",
        "axes.unicode_minus": False,
        "figure.autolayout":  False,   # we call tight_layout manually
    })
