"""Generate occupancy-grid and costmap diagrams for a planning lecture.

Three figures are produced from one shared 10x10 environment:

    1. occupancy_vs_costmap.png -- an occupancy grid (free / occupied) next to a
       costmap of the same scene.
    2. costmap_basic.png -- a costmap with low cost in free space and high cost
       on the obstacles.
    3. costmap_inflated.png -- the same costmap with an inflation layer, so cost
       rises smoothly near obstacles to keep planned paths at a safe distance.
"""

from pathlib import Path

import heapq

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
GRID_SIZE = 10

# Cost scale (loosely following ROS costmap conventions).
LETHAL_COST = 100
INFLATION_DECAY = 0.55  # how quickly the inflated cost falls off per cell

# Planning comparison: start, goal, and how strongly the costmap is weighted.
PLAN_START = (0, 4)
PLAN_GOAL = (9, 4)
INFLATION_WEIGHT = 0.6

OUTPUT_DIRECTORY = Path(__file__).resolve().parent

GRID_LINE_COLOR = "#B0B0B0"
FREE_COLOR = "#FFFFFF"
OCCUPIED_COLOR = "#252525"
COST_CMAP = "RdYlGn_r"  # green = low cost, red = high cost
PATH_COLOR = "#111111"
START_COLOR = "#E53935"
GOAL_COLOR = "#43A047"
DARK = "#252525"


def build_occupancy() -> np.ndarray:
    """Return a 10x10 occupancy grid (0 = free, 1 = occupied)."""
    occupancy = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)  # [row, column]

    # An L-shaped wall.
    occupancy[2:7, 2] = 1      # vertical arm
    occupancy[6, 2:7] = 1      # horizontal arm
    # A small block in the upper right.
    occupancy[2:4, 6:8] = 1

    return occupancy


def distance_to_obstacles(occupancy: np.ndarray) -> np.ndarray:
    """Return each cell's Euclidean distance to the nearest occupied cell."""
    obstacles = np.argwhere(occupancy == 1)
    rows, columns = np.indices(occupancy.shape)
    cells = np.stack([rows.ravel(), columns.ravel()], axis=1)
    deltas = cells[:, None, :] - obstacles[None, :, :]
    distances = np.sqrt((deltas**2).sum(axis=2)).min(axis=1)
    return distances.reshape(occupancy.shape)


def basic_costmap(occupancy: np.ndarray) -> np.ndarray:
    """Return a costmap: low cost in free space, lethal cost on obstacles."""
    return np.where(occupancy == 1, LETHAL_COST, 0).astype(float)


def inflated_costmap(occupancy: np.ndarray) -> np.ndarray:
    """Return a costmap whose cost decays exponentially away from obstacles."""
    distance = distance_to_obstacles(occupancy)
    inflated = (LETHAL_COST - 1) * np.exp(-INFLATION_DECAY * distance)
    inflated = np.clip(inflated, 0, LETHAL_COST - 1)
    return np.where(occupancy == 1, LETHAL_COST, inflated)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def draw_grid_lines(axis: plt.Axes) -> None:
    """Draw thin cell borders over a 20x20 panel."""
    for index in range(GRID_SIZE + 1):
        coordinate = index - 0.5
        axis.plot([-0.5, GRID_SIZE - 0.5], [coordinate, coordinate],
                  color=GRID_LINE_COLOR, linewidth=0.5, zorder=3)
        axis.plot([coordinate, coordinate], [-0.5, GRID_SIZE - 0.5],
                  color=GRID_LINE_COLOR, linewidth=0.5, zorder=3)


def style_axis(axis: plt.Axes, title: str) -> None:
    """Apply the shared square, unlabeled styling to a panel."""
    axis.set_xlim(-0.5, GRID_SIZE - 0.5)
    axis.set_ylim(-0.5, GRID_SIZE - 0.5)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_title(title, fontsize=15, weight="bold", color=DARK, pad=12)


def draw_occupancy(axis: plt.Axes, occupancy: np.ndarray, title: str) -> None:
    """Draw a binary occupancy grid (white free, black occupied)."""
    axis.imshow(
        occupancy, cmap=ListedColormap([FREE_COLOR, OCCUPIED_COLOR]),
        origin="lower", extent=(-0.5, GRID_SIZE - 0.5, -0.5, GRID_SIZE - 0.5),
        vmin=0, vmax=1, interpolation="nearest", zorder=1,
    )
    draw_grid_lines(axis)
    style_axis(axis, title)


def draw_costmap(axis: plt.Axes, cost: np.ndarray, title: str, show_values: bool = True):
    """Draw a costmap with the shared color scale and (optionally) per-cell values."""
    image = axis.imshow(
        cost, cmap=COST_CMAP, origin="lower",
        extent=(-0.5, GRID_SIZE - 0.5, -0.5, GRID_SIZE - 0.5),
        vmin=0, vmax=LETHAL_COST, interpolation="nearest", zorder=1,
    )
    draw_grid_lines(axis)

    # Numeric cost printed in every cell, in addition to the color scale.
    if show_values:
        for row in range(GRID_SIZE):
            for column in range(GRID_SIZE):
                value = int(round(cost[row, column]))
                axis.text(
                    column, row, str(value), ha="center", va="center",
                    fontsize=9, weight="bold", zorder=4,
                    color="white" if value >= 60 else DARK,
                )

    style_axis(axis, title)
    return image


def add_colorbar(figure: plt.Figure, axis: plt.Axes, image) -> None:
    """Attach a labeled cost colorbar to a panel."""
    bar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    bar.set_label("cost", fontsize=12, weight="bold")
    bar.set_ticks([0, LETHAL_COST // 2, LETHAL_COST])
    bar.ax.set_yticklabels(["low", "med", "high"])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def figure_occupancy_vs_costmap(occupancy: np.ndarray) -> None:
    """Figure 1: occupancy grid beside its costmap."""
    figure, (axis_left, axis_right) = plt.subplots(1, 2, figsize=(13, 6.5),
                                                   constrained_layout=True)
    draw_occupancy(axis_left, occupancy, "Occupancy grid\n(free / occupied)")
    image = draw_costmap(axis_right, basic_costmap(occupancy),
                         "Costmap\n(cost per cell)")
    add_colorbar(figure, axis_right, image)
    figure.suptitle("Occupancy grid vs. costmap", fontsize=19, weight="bold", color=DARK)
    save_figure(figure, "occupancy_vs_costmap")


def figure_basic_costmap(occupancy: np.ndarray) -> None:
    """Figure 2: a costmap with low cost in free space, high cost on obstacles."""
    figure, axis = plt.subplots(figsize=(7.5, 7.5), constrained_layout=True)
    image = draw_costmap(axis, basic_costmap(occupancy),
                         "Costmap: low cost free, high cost on obstacles")
    add_colorbar(figure, axis, image)
    save_figure(figure, "costmap_basic")


def figure_inflated_costmap(occupancy: np.ndarray) -> None:
    """Figure 3: the costmap with inflation for obstacle avoidance."""
    figure, axis = plt.subplots(figsize=(7.5, 7.5), constrained_layout=True)
    image = draw_costmap(axis, inflated_costmap(occupancy),
                         "Inflated costmap: cost rises near obstacles")
    add_colorbar(figure, axis, image)
    save_figure(figure, "costmap_inflated")


def plan_path(occupancy, cost_field, start, goal, weight):
    """Plan an 8-connected A* path; each step's cost adds weight * the cell cost."""
    def is_free(node):
        column, row = node
        return (0 <= column < GRID_SIZE and 0 <= row < GRID_SIZE
                and occupancy[row, column] == 0)

    def neighbors(node):
        column, row = node
        result = []
        for column_step in (-1, 0, 1):
            for row_step in (-1, 0, 1):
                if column_step == 0 and row_step == 0:
                    continue
                neighbor = (column + column_step, row + row_step)
                if not is_free(neighbor):
                    continue
                if column_step != 0 and row_step != 0 and (
                        not is_free((column + column_step, row))
                        or not is_free((column, row + row_step))):
                    continue
                result.append(neighbor)
        return result

    def heuristic(node):
        return 10 * np.hypot(node[0] - goal[0], node[1] - goal[1])

    open_heap = [(heuristic(start), 0, start)]
    cost_so_far = {start: 0.0}
    came_from = {}
    counter = 0
    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current == goal:
            break
        for neighbor in neighbors(current):
            diagonal = neighbor[0] != current[0] and neighbor[1] != current[1]
            step = (14 if diagonal else 10) + weight * cost_field[neighbor[1], neighbor[0]]
            new_cost = cost_so_far[current] + step
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                counter += 1
                heapq.heappush(open_heap, (new_cost + heuristic(neighbor), counter, neighbor))

    path = [goal]
    while path[-1] in came_from:
        path.append(came_from[path[-1]])
    return path[::-1]


def draw_path(axis: plt.Axes, path) -> None:
    """Draw a planned path as a dark line with a white casing for contrast."""
    xs = [column for column, _ in path]
    ys = [row for _, row in path]
    axis.plot(xs, ys, color="white", linewidth=5.5, zorder=5, solid_capstyle="round")
    axis.plot(xs, ys, color=PATH_COLOR, linewidth=3.0, zorder=6, solid_capstyle="round")


def draw_endpoints(axis: plt.Axes) -> None:
    """Draw the start (circle) and goal (star) with a white halo for contrast."""
    for position, color, marker, size in [
        (PLAN_START, START_COLOR, "o", 340),
        (PLAN_GOAL, GOAL_COLOR, "*", 480),
    ]:
        axis.scatter(*position, s=int(size * 1.7), marker=marker, color="white", zorder=6.5)
        axis.scatter(*position, s=size, marker=marker, facecolor=color,
                     edgecolor=DARK, linewidth=1.5, zorder=7)


def figure_planning_comparison(occupancy: np.ndarray) -> None:
    """Figure 4: A* without a costmap (hugs obstacles) vs with one (keeps clear)."""
    inflated = inflated_costmap(occupancy)
    zero_cost = np.zeros_like(inflated)
    hugging = plan_path(occupancy, zero_cost, PLAN_START, PLAN_GOAL, 0.0)
    clearance = plan_path(occupancy, inflated, PLAN_START, PLAN_GOAL, INFLATION_WEIGHT)

    figure, (axis_left, axis_right) = plt.subplots(1, 2, figsize=(13, 6.8),
                                                   constrained_layout=True)
    draw_occupancy(axis_left, occupancy, "No costmap:\nA* hugs the obstacles")
    draw_path(axis_left, hugging)
    draw_endpoints(axis_left)

    image = draw_costmap(axis_right, inflated,
                         "With costmap:\nA* keeps a safe distance", show_values=False)
    draw_path(axis_right, clearance)
    draw_endpoints(axis_right)
    add_colorbar(figure, axis_right, image)

    figure.suptitle("Planning with vs. without a costmap", fontsize=19,
                    weight="bold", color=DARK)
    save_figure(figure, "planning_with_costmap")


def save_figure(figure: plt.Figure, file_stem: str) -> None:
    """Save a figure as a slide-ready PNG file, then close it."""
    png_path = OUTPUT_DIRECTORY / f"{file_stem}.png"
    figure.savefig(png_path, dpi=300, facecolor="white")
    plt.close(figure)
    print(f"Saved {png_path}")


def main() -> None:
    occupancy = build_occupancy()
    figure_occupancy_vs_costmap(occupancy)
    figure_basic_costmap(occupancy)
    figure_inflated_costmap(occupancy)
    figure_planning_comparison(occupancy)


if __name__ == "__main__":
    main()
