"""Show why modeling the robot as a point can hide a collision.

A grid planner usually treats the robot as a single point at a cell center, so
the "optimal" path is free to cut diagonally past the corner of an obstacle. The
point never touches the obstacle -- but the robot's real chassis has width, so as
it follows that diagonal it clips the obstacle corner.

The figure draws an occupancy grid, the optimal diagonal path (which grazes the
obstacle's corner), the robot drawn as a point on that path, and the robot's
footprint as a circle superimposed at the corner-cut, overlapping the obstacle.
The overlap is highlighted to make the collision obvious.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
GRID_SIZE = 10

# The path is the main diagonal; it grazes the obstacle corner at this point.
START_NODE = (0, 0)
GOAL_NODE = (9, 9)
CORNER_CUT = (4.5, 4.5)  # midpoint of the (4,4) -> (5,5) diagonal step

# Robot footprint (in cell units): wider than one cell, so it cannot fit through
# the corner the point model happily cuts.
ROBOT_SIZE = 1.5

OUTPUT_DIRECTORY = Path(__file__).resolve().parent

GRID_LINE_COLOR = "#B0B0B0"
FREE_COLOR = "#FFFFFF"
OCCUPIED_COLOR = "#252525"
PATH_COLOR = "#1976D2"
ROBOT_COLOR = "#1976D2"
COLLISION_COLOR = "#E53935"
START_COLOR = "#E53935"
GOAL_COLOR = "#43A047"
DARK = "#252525"


def build_occupancy() -> np.ndarray:
    """Return a 10x10 occupancy grid with a block in the lower right."""
    occupancy = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)  # [row, column]
    occupancy[0:5, 5:10] = 1  # rows 0-4, columns 5-9
    return occupancy


def diagonal_path() -> list[tuple[int, int]]:
    """Return the optimal all-diagonal path from start to goal."""
    return [(i, i) for i in range(GRID_SIZE)]


def obstacle_bounding_box(occupancy: np.ndarray):
    """Return the (x0, y0, width, height) rectangle covering all obstacle cells."""
    rows, columns = np.where(occupancy == 1)
    x0, x1 = columns.min() - 0.5, columns.max() + 0.5
    y0, y1 = rows.min() - 0.5, rows.max() + 0.5
    return x0, y0, x1 - x0, y1 - y0


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_grid_lines(axis: plt.Axes) -> None:
    """Draw thin cell borders over the grid."""
    for index in range(GRID_SIZE + 1):
        coordinate = index - 0.5
        axis.plot([-0.5, GRID_SIZE - 0.5], [coordinate, coordinate],
                  color=GRID_LINE_COLOR, linewidth=0.6, zorder=2)
        axis.plot([coordinate, coordinate], [-0.5, GRID_SIZE - 0.5],
                  color=GRID_LINE_COLOR, linewidth=0.6, zorder=2)


def main() -> None:
    occupancy = build_occupancy()
    path = diagonal_path()

    figure, axis = plt.subplots(figsize=(8, 8))

    axis.imshow(
        occupancy, cmap=ListedColormap([FREE_COLOR, OCCUPIED_COLOR]),
        origin="lower", extent=(-0.5, GRID_SIZE - 0.5, -0.5, GRID_SIZE - 0.5),
        vmin=0, vmax=1, interpolation="nearest", zorder=1,
    )
    draw_grid_lines(axis)

    # Optimal path through the cell centers.
    axis.plot([c for c, _ in path], [r for _, r in path],
              color=PATH_COLOR, linewidth=3.0, zorder=3)
    axis.scatter([c for c, _ in path], [r for _, r in path],
                 s=60, color=PATH_COLOR, zorder=3)

    # Robot footprint (a circle) superimposed at the corner-cut.
    cx, cy = CORNER_CUT
    radius = ROBOT_SIZE / 2
    axis.add_patch(Circle((cx, cy), radius, facecolor=ROBOT_COLOR,
                          edgecolor=ROBOT_COLOR, alpha=0.30, linewidth=2.5, zorder=4))

    # Highlight the collision: the part of the footprint inside the obstacle.
    collision = Circle((cx, cy), radius, facecolor=COLLISION_COLOR,
                       edgecolor="none", alpha=0.75, zorder=5)
    axis.add_patch(collision)
    bx0, by0, bw, bh = obstacle_bounding_box(occupancy)
    collision.set_clip_path(Rectangle((bx0, by0), bw, bh, transform=axis.transData))

    # Footprint outline and a heading arrow make it read as a little robot.
    axis.add_patch(Circle((cx, cy), radius, facecolor="none",
                          edgecolor=ROBOT_COLOR, linewidth=2.5, zorder=6))
    heading = radius * 0.92 / np.sqrt(2)
    axis.annotate("", xy=(cx + heading, cy + heading), xytext=(cx, cy),
                  arrowprops=dict(arrowstyle="-|>", color="#0D47A1", linewidth=2.5),
                  zorder=7)

    # The robot modeled as a point at its center.
    axis.scatter(cx, cy, s=90, marker="o", facecolor="white",
                 edgecolor=DARK, linewidth=2.0, zorder=7)

    # Start and goal markers.
    axis.scatter(*START_NODE, s=360, marker="o", facecolor=START_COLOR,
                 edgecolor="white", linewidth=2.0, zorder=7)
    axis.scatter(*GOAL_NODE, s=440, marker="*", facecolor=GOAL_COLOR,
                 edgecolor="white", linewidth=1.8, zorder=7)

    axis.set_xlim(-0.5, GRID_SIZE - 0.5)
    axis.set_ylim(-0.5, GRID_SIZE - 0.5)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_title("Point-robot model vs. real footprint", fontsize=18,
                   weight="bold", color=DARK, pad=12)

    legend_handles = [
        Line2D([0], [0], color=PATH_COLOR, linewidth=3, label="Optimal path (point robot)"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white",
               markeredgecolor=DARK, markersize=9, label="Robot as a point"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=ROBOT_COLOR,
               markeredgecolor=ROBOT_COLOR, markersize=15, alpha=0.5,
               label="Robot footprint"),
        Patch(facecolor=COLLISION_COLOR, alpha=0.75, edgecolor="#B71C1C",
              label="Footprint overlaps obstacle"),
    ]
    axis.legend(handles=legend_handles, loc="upper center",
                bbox_to_anchor=(0.5, -0.03), ncol=2, fontsize=11, framealpha=0.95)

    axis.text(
        0.5, -0.19,
        "The planner cuts the corner because the point never hits the obstacle "
        "\u2014 but the robot's body does.",
        transform=axis.transAxes, fontsize=12.5, weight="bold", color=DARK,
        ha="center", va="top",
    )

    png_path = OUTPUT_DIRECTORY / "point_robot_collision.png"
    figure.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
