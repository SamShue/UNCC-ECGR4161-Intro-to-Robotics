"""Grow an RRT through the occupancy-grid floor plan and draw the resulting tree.

The map is the same floor plan used by the occupancy-grid diagram. An RRT is
grown from the robot toward the goal: it samples random points, steers the
nearest tree node a short step toward each sample, and keeps the step only if it
stays in free space. The whole tree of branches is drawn over the map, with the
solution path from the robot to the goal highlighted.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
GRID_SIZE = 20
ROBOT_POSITION = (3, 4)     # (column, row)
GOAL_POSITION = (16, 15)

STEP_SIZE = 2.5             # each extension jumps a few grid cells
GOAL_BIAS = 0.10           # fraction of samples aimed straight at the goal
GOAL_THRESHOLD = 2.0       # how close a node must be to connect to the goal
MAX_NODES = 150            # far fewer nodes than the map has free cells
COLLISION_RESOLUTION = 0.15
SEED = 12

OUTPUT_DIRECTORY = Path(__file__).resolve().parent

FREE_COLOR = "#FFFFFF"
OCCUPIED_COLOR = "#252525"
GRID_LINE_COLOR = "#D0D0D0"
TREE_COLOR = "#64B5F6"
NODE_COLOR = "#1E88E5"
PATH_COLOR = "#FB8C00"
ROBOT_COLOR = "#E53935"
GOAL_COLOR = "#1565C0"
DARK = "#252525"


def create_floor_plan(size: int = GRID_SIZE) -> np.ndarray:
    """Return the occupancy grid floor plan (0 = free, 1 = occupied)."""
    grid = np.zeros((size, size), dtype=np.uint8)

    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1

    main_wall_column = 9
    grid[1:-1, main_wall_column] = 1
    grid[5:8, main_wall_column] = 0

    upper_room_wall_row = 11
    grid[upper_room_wall_row, main_wall_column:-1] = 1
    grid[upper_room_wall_row, 15:17] = 0

    grid[12, 1:7] = 1
    grid[12:16, 6] = 1
    grid[12, 4:6] = 0

    grid[2:4, 13:16] = 1
    grid[15:17, 11:13] = 1

    return grid


# ---------------------------------------------------------------------------
# RRT
# ---------------------------------------------------------------------------
def is_free(point: np.ndarray, grid: np.ndarray) -> bool:
    """Return True if the cell containing a point is inside the map and free."""
    column = int(round(point[0]))
    row = int(round(point[1]))
    if 0 <= column < GRID_SIZE and 0 <= row < GRID_SIZE:
        return grid[row, column] == 0
    return False


def edge_is_free(a: np.ndarray, b: np.ndarray, grid: np.ndarray) -> bool:
    """Return True if the straight segment a-b stays in free space."""
    distance = float(np.hypot(*(b - a)))
    steps = max(1, int(distance / COLLISION_RESOLUTION))
    for index in range(steps + 1):
        point = a + (b - a) * (index / steps)
        if not is_free(point, grid):
            return False
    return True


def grow_rrt(grid: np.ndarray):
    """Grow an RRT; return its nodes, parent links, and the goal node index."""
    rng = np.random.default_rng(SEED)
    start = np.array(ROBOT_POSITION, dtype=float)
    goal = np.array(GOAL_POSITION, dtype=float)

    nodes = [start]
    parents = [-1]
    goal_index = None
    occupied = {(int(start[0]), int(start[1]))}

    while len(nodes) < MAX_NODES:
        if rng.random() < GOAL_BIAS:
            sample = goal
        else:
            sample = rng.uniform(1.0, GRID_SIZE - 2.0, size=2)

        node_array = np.array(nodes)
        distances = np.hypot(node_array[:, 0] - sample[0], node_array[:, 1] - sample[1])
        nearest_index = int(distances.argmin())
        nearest = nodes[nearest_index]

        direction = sample - nearest
        length = float(np.hypot(*direction))
        if length < 1e-9:
            continue
        steered = nearest + direction / length * min(STEP_SIZE, length)

        # Snap each new node to a grid cell so it sits inside one distinct cell.
        new_node = np.round(steered)
        cell = (int(new_node[0]), int(new_node[1]))
        if cell in occupied:
            continue
        if not is_free(new_node, grid) or not edge_is_free(nearest, new_node, grid):
            continue

        nodes.append(new_node)
        parents.append(nearest_index)
        occupied.add(cell)

        if goal_index is None and np.hypot(*(new_node - goal)) <= GOAL_THRESHOLD \
                and edge_is_free(new_node, goal, grid):
            nodes.append(goal.copy())
            parents.append(len(nodes) - 2)
            goal_index = len(nodes) - 1

    return nodes, parents, goal_index


def reconstruct_path(nodes, parents, goal_index):
    """Return the node coordinates along the path from the goal back to start."""
    if goal_index is None:
        return None
    path = []
    index = goal_index
    while index != -1:
        path.append(nodes[index])
        index = parents[index]
    return np.array(path[::-1])


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def main() -> None:
    grid = create_floor_plan()
    nodes, parents, goal_index = grow_rrt(grid)
    path = reconstruct_path(nodes, parents, goal_index)

    figure, axis = plt.subplots(figsize=(8, 8))

    axis.imshow(grid, cmap=ListedColormap([FREE_COLOR, OCCUPIED_COLOR]),
                origin="lower", extent=(-0.5, GRID_SIZE - 0.5, -0.5, GRID_SIZE - 0.5),
                vmin=0, vmax=1, interpolation="nearest", zorder=1)

    for index in range(GRID_SIZE + 1):
        coordinate = index - 0.5
        axis.plot([-0.5, GRID_SIZE - 0.5], [coordinate, coordinate],
                  color=GRID_LINE_COLOR, linewidth=0.5, zorder=2)
        axis.plot([coordinate, coordinate], [-0.5, GRID_SIZE - 0.5],
                  color=GRID_LINE_COLOR, linewidth=0.5, zorder=2)

    # RRT tree edges as a single line collection.
    segments = [[nodes[parents[i]], nodes[i]] for i in range(1, len(nodes))
                if parents[i] != -1]
    axis.add_collection(LineCollection(segments, colors=TREE_COLOR,
                                       linewidths=0.9, zorder=3))
    node_array = np.array(nodes)
    axis.scatter(node_array[:, 0], node_array[:, 1], s=5, color=NODE_COLOR, zorder=4)

    # Highlighted solution path.
    if path is not None:
        axis.plot(path[:, 0], path[:, 1], color=PATH_COLOR, linewidth=3.5, zorder=5)

    # Robot and goal.
    axis.scatter(*ROBOT_POSITION, s=430, marker="o", facecolor=ROBOT_COLOR,
                 edgecolor="white", linewidth=2.0, zorder=6)
    axis.scatter(*GOAL_POSITION, s=560, marker="*", facecolor=GOAL_COLOR,
                 edgecolor="white", linewidth=1.8, zorder=6)

    axis.set_xlim(-0.5, GRID_SIZE - 0.5)
    axis.set_ylim(-0.5, GRID_SIZE - 0.5)
    axis.set_aspect("equal")
    axis.axis("off")

    legend_handles = [
        Line2D([0], [0], color=TREE_COLOR, linewidth=1.5, label="RRT tree"),
        Line2D([0], [0], color=PATH_COLOR, linewidth=3, label="Path to goal"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=ROBOT_COLOR,
               markeredgecolor="white", markersize=12, label="Robot"),
        Line2D([0], [0], marker="*", linestyle="none", markerfacecolor=GOAL_COLOR,
               markeredgecolor="white", markersize=15, label="Goal"),
    ]
    axis.legend(handles=legend_handles, loc="upper center", ncol=4,
                bbox_to_anchor=(0.5, -0.02), fontsize=11, framealpha=0.95)

    png_path = OUTPUT_DIRECTORY / "rrt_on_floor_plan.png"
    figure.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {png_path}")
    print(f"RRT nodes: {len(nodes)}   free cells: {int((grid == 0).sum())}")
    if goal_index is None:
        print("Note: the tree did not connect to the goal; try more nodes or a larger step.")


if __name__ == "__main__":
    main()
