"""Half-court drawing in the project's shot frame: feet from the basket, folded |dx|."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Arc, Circle, Rectangle

LINE = "#52514e"


def draw_folded_half_court(
    ax: Axes, three_ft: float = 22.146, corner_ft: float = 21.65, lw: float = 0.9
) -> None:
    """Right half of a half court (dx >= 0), basket at the origin, baseline at dy = -5.25."""
    baseline = -5.25
    ax.add_patch(Circle((0, 0), 0.75, fill=False, color=LINE, lw=lw))
    ax.plot([0, 3], [-1.25, -1.25], color=LINE, lw=lw)  # backboard (half)
    ax.add_patch(Rectangle((0, baseline), 6, 19, fill=False, color=LINE, lw=lw))  # lane (half)
    ax.add_patch(Arc((0, 13.75), 12, 12, theta1=0, theta2=90, color=LINE, lw=lw))
    ax.add_patch(Arc((0, 0), 8, 8, theta1=0, theta2=90, color=LINE, lw=lw))  # restricted area
    # Three-point line: straight corner segment, then the arc.
    theta = np.degrees(np.arccos(corner_ft / three_ft))
    corner_top = three_ft * np.sin(np.radians(theta))
    ax.plot([corner_ft, corner_ft], [baseline, corner_top], color=LINE, lw=lw)
    ax.add_patch(
        Arc((0, 0), 2 * three_ft, 2 * three_ft, theta1=theta, theta2=90, color=LINE, lw=lw)
    )
    ax.plot([0, 25], [baseline, baseline], color=LINE, lw=lw)
    ax.plot([25, 25], [baseline, 30], color=LINE, lw=lw)
    ax.set_xlim(-0.5, 25.5)
    ax.set_ylim(baseline - 0.5, 30)
    ax.set_aspect("equal")
    ax.axis("off")
