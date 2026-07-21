"""Illustrate why RRT limits how far each new node may be placed.

Three images tell the story:
    1. A random sample places a candidate node on the far side of a wall.
    2. The edge back to the tree crosses the wall, so the node is discarded.
    3. Capping the step length keeps new nodes close to the tree, so the
       connecting edge usually stays clear of walls.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
BOUNDS = 10.0
WALL_X = (5.0, 5.6)
WALL_Y = (1.8, 8.2)

ROOT = np.array([1.3, 6.6])
NEAREST = np.array([2.6, 5.0])
OTHER = np.array([1.6, 3.4])
EDGES = [(ROOT, NEAREST), (ROOT, OTHER)]

SAMPLE = np.array([8.6, 5.4])
STEP = 1.2  # the distance cap used in the third image

OUTPUT_DIRECTORY = Path(__file__).resolve().parent

BORDER_COLOR = "#607D8B"
WALL_COLOR = "#252525"
TREE_COLOR = "#1976D2"
NEW_COLOR = "#FB8C00"
DISCARD_COLOR = "#C62828"
VALID_COLOR = "#2E7D32"
SAMPLE_COLOR = "#616161"
DARK = "#252525"


def limited_node() -> np.ndarray:
    """Return the node produced by stepping at most STEP toward the sample."""
    direction = SAMPLE - NEAREST
    return NEAREST + direction / np.hypot(*direction) * STEP


def wall_crossing() -> np.ndarray:
    """Return where the nearest-to-sample segment meets the wall's near face."""
    t = (WALL_X[0] - NEAREST[0]) / (SAMPLE[0] - NEAREST[0])
    return np.array([WALL_X[0], NEAREST[1] + t * (SAMPLE[1] - NEAREST[1])])


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def create_axes() -> tuple[plt.Figure, plt.Axes]:
    """Return a figure and axes with the environment and wall drawn."""
    figure, axis = plt.subplots(figsize=(7.2, 7.4))
    axis.add_patch(Rectangle((0, 0), BOUNDS, BOUNDS, facecolor="#FAFAFA",
                             edgecolor=BORDER_COLOR, linewidth=2.0, zorder=0))
    axis.add_patch(Rectangle((WALL_X[0], WALL_Y[0]), WALL_X[1] - WALL_X[0],
                             WALL_Y[1] - WALL_Y[0], facecolor=WALL_COLOR,
                             edgecolor=WALL_COLOR, zorder=1))
    axis.text((WALL_X[0] + WALL_X[1]) / 2, WALL_Y[1] - 0.6, "wall", rotation=90,
              ha="center", va="top", color="white", fontsize=12, weight="bold", zorder=2)

    for a, b in EDGES:
        axis.plot([a[0], b[0]], [a[1], b[1]], color=TREE_COLOR, linewidth=2.2, zorder=3)
    for node in (ROOT, NEAREST, OTHER):
        axis.scatter(*node, s=120, color=TREE_COLOR, edgecolor="white",
                     linewidth=1.5, zorder=4)
    axis.text(NEAREST[0], NEAREST[1] - 0.55, "nearest node", ha="center", va="top",
              fontsize=11, weight="bold", color=TREE_COLOR, zorder=5)

    axis.set_xlim(-0.4, BOUNDS + 0.4)
    axis.set_ylim(-0.4, BOUNDS + 0.4)
    axis.set_aspect("equal")
    axis.axis("off")
    return figure, axis


def draw_sample(axis) -> None:
    """Draw the random sample marker and label."""
    axis.scatter(*SAMPLE, s=150, marker="x", color=SAMPLE_COLOR,
                 linewidth=2.5, zorder=6)
    axis.text(SAMPLE[0], SAMPLE[1] + 0.5, "sample", ha="center", va="bottom",
              fontsize=11, weight="bold", color=SAMPLE_COLOR, zorder=6)


def caption(axis, text, color=DARK) -> None:
    """Place a caption below the scene."""
    axis.text(0.5, -0.03, text, transform=axis.transAxes, fontsize=13,
              weight="bold", color=color, ha="center", va="top")


def save(figure, stem) -> None:
    png_path = OUTPUT_DIRECTORY / f"{stem}.png"
    figure.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {png_path}")


def render_generated() -> None:
    """Image 1: a candidate node is generated on the far side of the wall."""
    figure, axis = create_axes()
    draw_sample(axis)
    axis.annotate("", xy=SAMPLE, xytext=NEAREST, zorder=5, arrowprops=dict(
        arrowstyle="-|>", color=NEW_COLOR, linewidth=2.5, linestyle="--"))
    axis.scatter(*SAMPLE, s=200, color=NEW_COLOR, edgecolor="white",
                 linewidth=1.8, zorder=7)
    axis.text(SAMPLE[0], SAMPLE[1] - 0.55, "new node", ha="center", va="top",
              fontsize=11, weight="bold", color=NEW_COLOR, zorder=7)
    caption(axis, "A new node is generated on the far side of the wall")
    save(figure, "wall_step_1_generated")


def render_discarded() -> None:
    """Image 2: the connecting edge crosses the wall, so the node is discarded."""
    figure, axis = create_axes()
    draw_sample(axis)
    axis.plot([NEAREST[0], SAMPLE[0]], [NEAREST[1], SAMPLE[1]],
              color=DISCARD_COLOR, linewidth=2.5, linestyle="--", zorder=5)
    axis.scatter(*SAMPLE, s=200, facecolor="none", edgecolor=DISCARD_COLOR,
                 linewidth=2.2, zorder=7)
    axis.scatter(*SAMPLE, s=120, marker="x", color=DISCARD_COLOR,
                 linewidth=2.5, zorder=8)
    crossing = wall_crossing()
    axis.scatter(*crossing, s=320, marker="X", color=DISCARD_COLOR,
                 edgecolor="white", linewidth=1.5, zorder=8)
    axis.text(crossing[0] + 0.25, crossing[1] + 0.45, "collision", ha="left",
              va="bottom", fontsize=11, weight="bold", color=DISCARD_COLOR, zorder=8)
    caption(axis, "The edge crosses the wall \u2014 the node is discarded", DISCARD_COLOR)
    save(figure, "wall_step_2_discarded")


def render_limited() -> None:
    """Image 3: capping the step keeps the new node near the tree."""
    figure, axis = create_axes()
    axis.add_patch(Circle(NEAREST, STEP, facecolor="none", edgecolor="#9E9E9E",
                          linestyle="--", linewidth=1.5, zorder=3))
    axis.text(NEAREST[0], NEAREST[1] + STEP + 0.2, "max step", ha="center",
              va="bottom", fontsize=10.5, weight="bold", color="#616161", zorder=5)
    draw_sample(axis)

    new_node = limited_node()
    axis.annotate("", xy=new_node, xytext=NEAREST, zorder=5, arrowprops=dict(
        arrowstyle="-|>", color=VALID_COLOR, linewidth=2.8))
    axis.plot([new_node[0], SAMPLE[0]], [new_node[1], SAMPLE[1]],
              color=SAMPLE_COLOR, linewidth=1.2, linestyle=":", zorder=4)
    axis.scatter(*new_node, s=200, color=VALID_COLOR, edgecolor="white",
                 linewidth=1.8, zorder=7)
    axis.text(new_node[0], new_node[1] + 0.5, "new node", ha="center", va="bottom",
              fontsize=11, weight="bold", color=VALID_COLOR, zorder=7)
    caption(axis, "Cap the step so new nodes stay near the tree and clear the wall",
            VALID_COLOR)
    save(figure, "wall_step_3_limited")


def main() -> None:
    render_generated()
    render_discarded()
    render_limited()


if __name__ == "__main__":
    main()
