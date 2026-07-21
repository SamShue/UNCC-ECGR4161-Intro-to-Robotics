"""Show why search-based planners scale poorly as grid resolution grows.

Two panels of the same physical size present the identical planning problem -- a
start in one corner and a goal in the opposite one -- discretized first as a
20x20 grid and then as a 1000x1000 grid. A uniform-cost search expands its
frontier outward from the start (shown as the shaded quarter-disc), so the
number of nodes it must expand grows with the number of cells.

The point is the density: refining the grid from 20x20 to 1000x1000 keeps the
same problem but multiplies the cells (and therefore the search effort) by
2,500x. On the left you can count the cells; on the right they collapse into a
solid mesh.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
COARSE_SIZE = 20
FINE_SIZE = 1000

OUTPUT_DIRECTORY = Path(__file__).resolve().parent

FREE_COLOR = "#FFFFFF"
EXPLORED_COLOR = "#BBDEFB"
FRONTIER_COLOR = "#1976D2"
START_COLOR = "#E53935"
GOAL_COLOR = "#43A047"
DARK = "#252525"


def explored_mask(size: int) -> np.ndarray:
    """Return a boolean grid of cells a uniform-cost search has expanded.

    The frontier is a quarter-disc centered on the start corner with a radius of
    one grid side, so it fills roughly pi/4 of the grid and has not yet reached
    the far goal corner -- a snapshot of a search still in progress.
    """
    columns, rows = np.meshgrid(np.arange(size), np.arange(size))
    distance_from_start = np.hypot(columns, rows)
    return distance_from_start <= (size - 1)


def draw_panel(axis: plt.Axes, size: int) -> None:
    """Draw one grid panel with its expanded frontier and start/goal markers."""
    explored = explored_mask(size)

    axis.imshow(
        explored,
        cmap=ListedColormap([FREE_COLOR, EXPLORED_COLOR]),
        origin="lower",
        extent=(-0.5, size - 0.5, -0.5, size - 0.5),
        vmin=0,
        vmax=1,
        interpolation="nearest",
        zorder=1,
    )

    # Grid lines: crisp and countable when coarse, a dense mesh when fine.
    line_positions = np.arange(-0.5, size, 1.0)
    if size <= 50:
        line_width, line_color, line_alpha = 0.9, "#4F4F4F", 1.0
    else:
        line_width, line_color, line_alpha = 0.25, "#8C8C8C", 0.6
    axis.vlines(line_positions, -0.5, size - 0.5, colors=line_color,
                linewidth=line_width, alpha=line_alpha, zorder=2)
    axis.hlines(line_positions, -0.5, size - 0.5, colors=line_color,
                linewidth=line_width, alpha=line_alpha, zorder=2)

    # The search frontier: a quarter-circle boundary of the expanded region.
    angles = np.linspace(0, np.pi / 2, 400)
    radius = size - 1
    axis.plot(radius * np.cos(angles), radius * np.sin(angles),
              color=FRONTIER_COLOR, linewidth=3.0, zorder=3)

    # Start and goal markers, kept the same visual size in both panels.
    axis.scatter(0, 0, s=520, marker="o", facecolor=START_COLOR,
                 edgecolor="white", linewidth=2.5, zorder=4)
    axis.scatter(size - 1, size - 1, s=620, marker="*", facecolor=GOAL_COLOR,
                 edgecolor="white", linewidth=2.0, zorder=4)

    axis.set_xlim(-0.5, size - 0.5)
    axis.set_ylim(-0.5, size - 0.5)
    axis.set_aspect("equal")
    axis.axis("off")

    expanded = int(explored.sum())
    axis.set_title(
        f"{size}\u00d7{size} grid \u2014 {size * size:,} cells\n"
        f"search has expanded ~{expanded:,} nodes",
        fontsize=15, weight="bold", color=DARK, pad=14,
    )


def main() -> None:
    figure, (axis_coarse, axis_fine) = plt.subplots(1, 2, figsize=(14.5, 8.2))

    draw_panel(axis_coarse, COARSE_SIZE)
    draw_panel(axis_fine, FINE_SIZE)

    ratio = (FINE_SIZE * FINE_SIZE) // (COARSE_SIZE * COARSE_SIZE)
    figure.suptitle(
        "Search-based planners scale poorly with resolution",
        fontsize=20, weight="bold", color=DARK, y=0.98,
    )
    figure.text(
        0.5, 0.05,
        f"Same start and goal, finer grid: the search space grows from "
        f"{COARSE_SIZE * COARSE_SIZE:,} to {FINE_SIZE * FINE_SIZE:,} cells "
        f"\u2014 {ratio:,}\u00d7 more nodes to expand.",
        fontsize=14, color=DARK, ha="center", va="center",
    )

    figure.subplots_adjust(left=0.03, right=0.97, top=0.86, bottom=0.12, wspace=0.08)

    png_path = OUTPUT_DIRECTORY / "search_scalability.png"
    figure.savefig(png_path, dpi=300, facecolor="white")
    plt.close(figure)
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
