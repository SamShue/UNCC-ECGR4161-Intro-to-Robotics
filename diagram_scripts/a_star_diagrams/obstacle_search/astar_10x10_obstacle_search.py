"""Generate an A* teaching sequence on a 10x10 map with a horizontal obstacle.

This mirrors ``astar_10x10_search.py`` but places a horizontal wall across the
middle of the map so the direct diagonal route to the goal is blocked. A* must
now search around the wall, which is exactly the behavior the sequence shows.

Costs are the same as the obstacle-free version:
    g = accumulated edge cost from the start (10 orthogonal, 14 diagonal)
    h = straight-line (Euclidean) distance from a node to the goal, scaled so
        that one cell equals 10 units and rounded to an integer
    f = g + h = the value A* uses to choose which node to expand next

Image sequence:
    1. The grid and graph nodes with a red start, green goal, and black wall.
    2. One node with a line drawn to the goal, labeled with its distance h.
    3. That same node labeled with f = g + h.
    4. Every neighbor of the start labeled with its own f = g + h value.
    5+. Incremental A* expansion frames that march around the wall to the goal.
        Nodes committed to the path are highlighted as red circles.
"""

import heapq
from math import hypot
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ---------------------------------------------------------------------------
# Map settings
# ---------------------------------------------------------------------------
GRID_SIZE = 10

# Coordinates use the form (column, row), with (0, 0) at the lower-left.
START_NODE = (1, 1)
GOAL_NODE = (8, 8)

# The example node used to introduce h and f in images 2 and 3.
EXAMPLE_NODE = (2, 2)

# A horizontal wall spanning the middle of the map. Openings remain on both
# ends of the row, forcing A* to detour around one side to reach the goal.
OBSTACLE_ROW = 5
OBSTACLE_COLUMNS = range(2, 8)
OBSTACLE_CELLS = frozenset((column, OBSTACLE_ROW) for column in OBSTACLE_COLUMNS)

ORTHOGONAL_COST = 10
DIAGONAL_COST = 14
CELL_UNITS = 10  # One grid cell equals this many distance units.

# Output files are written next to this script.
OUTPUT_DIRECTORY = Path(__file__).resolve().parent

# Shared colors so the whole sequence looks continuous.
GRID_LINE_COLOR = "#252525"
OBSTACLE_COLOR = "#252525"
EDGE_COLOR = "#B0BEC5"
NODE_FACE_COLOR = "#CFD8DC"
NODE_EDGE_COLOR = "#607D8B"
START_COLOR = "#E53935"
GOAL_COLOR = "#43A047"
PATH_COLOR = "#E53935"
HEURISTIC_COLOR = "#1976D2"
G_LABEL_COLOR = "#00695C"
H_LABEL_COLOR = "#1976D2"
F_LABEL_COLOR = "#252525"


def is_free(node: tuple[int, int]) -> bool:
    """Return True if a cell is inside the map and not part of the wall."""
    column, row = node
    in_bounds = 0 <= column < GRID_SIZE and 0 <= row < GRID_SIZE
    return in_bounds and node not in OBSTACLE_CELLS


def neighbors(node: tuple[int, int]) -> list[tuple[int, int]]:
    """Return the free eight-connected neighbors of a cell.

    Diagonal moves that would clip the corner of the wall are rejected so the
    path never passes through a blocked corner.
    """
    column, row = node
    result = []
    for column_step in (-1, 0, 1):
        for row_step in (-1, 0, 1):
            if column_step == 0 and row_step == 0:
                continue
            neighbor = (column + column_step, row + row_step)
            if not is_free(neighbor):
                continue
            if column_step != 0 and row_step != 0:
                if not is_free((column + column_step, row)) or not is_free(
                    (column, row + row_step)
                ):
                    continue
            result.append(neighbor)
    return result


def edge_cost(node_a: tuple[int, int], node_b: tuple[int, int]) -> int:
    """Return 14 for a diagonal step and 10 for an orthogonal step."""
    is_diagonal = node_a[0] != node_b[0] and node_a[1] != node_b[1]
    return DIAGONAL_COST if is_diagonal else ORTHOGONAL_COST


def heuristic(node: tuple[int, int]) -> int:
    """Return the rounded straight-line distance from a node to the goal."""
    distance = hypot(GOAL_NODE[0] - node[0], GOAL_NODE[1] - node[1])
    return round(CELL_UNITS * distance)


def find_path() -> list[tuple[int, int]]:
    """Return the least-cost path from start to goal using A*."""
    open_heap: list[tuple[int, tuple[int, int]]] = [
        (heuristic(START_NODE), START_NODE)
    ]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    cost_so_far: dict[tuple[int, int], int] = {START_NODE: 0}

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current == GOAL_NODE:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for neighbor in neighbors(current):
            new_cost = cost_so_far[current] + edge_cost(current, neighbor)
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor)
                heapq.heappush(open_heap, (priority, neighbor))
                came_from[neighbor] = current

    raise ValueError("No path exists between the start and the goal.")


def cumulative_g(path: list[tuple[int, int]]) -> list[int]:
    """Return the accumulated g cost at each node along a path."""
    g_values = [0]
    for previous, current in zip(path, path[1:]):
        g_values.append(g_values[-1] + edge_cost(previous, current))
    return g_values


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def create_axes() -> tuple[plt.Figure, plt.Axes]:
    """Return a figure and axes prepared with a square, unlabeled canvas."""
    figure, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
    axis.set_xlim(-0.5, GRID_SIZE - 0.5)
    axis.set_ylim(-0.5, GRID_SIZE - 0.5)
    axis.set_aspect("equal")
    axis.axis("off")
    return figure, axis


def draw_grid(axis: plt.Axes) -> None:
    """Draw the 10x10 grid with black cell borders and a white interior."""
    axis.set_facecolor("white")
    for line_index in range(GRID_SIZE + 1):
        coordinate = line_index - 0.5
        axis.plot(
            [-0.5, GRID_SIZE - 0.5],
            [coordinate, coordinate],
            color=GRID_LINE_COLOR,
            linewidth=1.6,
            zorder=1,
        )
        axis.plot(
            [coordinate, coordinate],
            [-0.5, GRID_SIZE - 0.5],
            color=GRID_LINE_COLOR,
            linewidth=1.6,
            zorder=1,
        )


def draw_obstacle(axis: plt.Axes) -> None:
    """Fill the wall cells in black."""
    for column, row in OBSTACLE_CELLS:
        axis.add_patch(
            Rectangle(
                (column - 0.5, row - 0.5),
                1,
                1,
                facecolor=OBSTACLE_COLOR,
                edgecolor=OBSTACLE_COLOR,
                zorder=1.5,
            )
        )


def draw_all_nodes(axis: plt.Axes) -> None:
    """Draw a faint node at the center of every free cell."""
    free_cells = [
        (column, row)
        for column in range(GRID_SIZE)
        for row in range(GRID_SIZE)
        if is_free((column, row))
    ]
    axis.scatter(
        [column for column, _ in free_cells],
        [row for _, row in free_cells],
        s=240,
        marker="o",
        facecolor=NODE_FACE_COLOR,
        edgecolor=NODE_EDGE_COLOR,
        linewidth=1.2,
        zorder=2,
    )


def highlight_node(
    axis: plt.Axes, node: tuple[int, int], color: str, size: int = 320
) -> None:
    """Draw a single filled circle over a cell to emphasize it."""
    axis.scatter(
        node[0],
        node[1],
        s=size,
        marker="o",
        facecolor=color,
        edgecolor="white",
        linewidth=2.0,
        zorder=5,
    )


def draw_start_and_goal(axis: plt.Axes) -> None:
    """Highlight the start node in red and the goal node in green."""
    highlight_node(axis, START_NODE, START_COLOR)
    highlight_node(axis, GOAL_NODE, GOAL_COLOR)


def label_f_value(axis: plt.Axes, node: tuple[int, int], f_value: int) -> None:
    """Write a node's f value in a small box just above the node."""
    axis.text(
        node[0],
        node[1] + 0.34,
        str(f_value),
        fontsize=9,
        weight="bold",
        color=F_LABEL_COLOR,
        ha="center",
        va="center",
        zorder=6,
        bbox=dict(
            boxstyle="round,pad=0.16",
            facecolor="white",
            edgecolor=F_LABEL_COLOR,
            linewidth=1.0,
        ),
    )


def label_edge_g(
    axis: plt.Axes, node_a: tuple[int, int], node_b: tuple[int, int]
) -> None:
    """Draw the edge from node_a to node_b and label it with its cost g."""
    axis.plot(
        [node_a[0], node_b[0]],
        [node_a[1], node_b[1]],
        color=EDGE_COLOR,
        linewidth=2.0,
        zorder=3,
    )
    midpoint_column = (node_a[0] + node_b[0]) / 2
    midpoint_row = (node_a[1] + node_b[1]) / 2
    axis.text(
        midpoint_column,
        midpoint_row,
        str(edge_cost(node_a, node_b)),
        fontsize=8,
        weight="bold",
        color=G_LABEL_COLOR,
        ha="center",
        va="center",
        zorder=6,
        bbox=dict(
            boxstyle="round,pad=0.12",
            facecolor="white",
            edgecolor=G_LABEL_COLOR,
            linewidth=0.9,
        ),
    )


def draw_heuristic_line(axis: plt.Axes, node: tuple[int, int]) -> None:
    """Draw a dashed line from a node to the goal, labeled with distance h."""
    axis.plot(
        [node[0], GOAL_NODE[0]],
        [node[1], GOAL_NODE[1]],
        color=HEURISTIC_COLOR,
        linewidth=2.0,
        linestyle="--",
        zorder=4,
    )
    midpoint_column = (node[0] + GOAL_NODE[0]) / 2
    midpoint_row = (node[1] + GOAL_NODE[1]) / 2
    axis.text(
        midpoint_column + 0.35,
        midpoint_row,
        f"h = {heuristic(node)}",
        fontsize=11,
        weight="bold",
        color=H_LABEL_COLOR,
        ha="left",
        va="center",
        zorder=6,
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="white",
            edgecolor=H_LABEL_COLOR,
            linewidth=1.1,
        ),
    )


def draw_committed_path(axis: plt.Axes, path_so_far: list[tuple[int, int]]) -> None:
    """Draw the committed path as a red line with red circular nodes."""
    if len(path_so_far) > 1:
        axis.plot(
            [column for column, _ in path_so_far],
            [row for _, row in path_so_far],
            color=PATH_COLOR,
            linewidth=3.0,
            zorder=4,
        )
    for node in path_so_far:
        highlight_node(axis, node, PATH_COLOR)


def annotate(axis: plt.Axes, text: str) -> None:
    """Place an explanatory caption below the grid."""
    axis.text(
        0.5,
        -0.04,
        text,
        transform=axis.transAxes,
        fontsize=13,
        color=F_LABEL_COLOR,
        ha="center",
        va="top",
    )


def save_figure(figure: plt.Figure, file_stem: str) -> None:
    """Save a figure as a slide-ready PNG file, then close it."""
    png_path = OUTPUT_DIRECTORY / f"{file_stem}.png"
    figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Saved {png_path}")


# ---------------------------------------------------------------------------
# Image renderers
# ---------------------------------------------------------------------------
def render_start_and_goal() -> None:
    """Image 1: the grid, wall, and the start (red) and goal (green) nodes."""
    figure, axis = create_axes()
    draw_grid(axis)
    draw_obstacle(axis)
    draw_all_nodes(axis)
    draw_start_and_goal(axis)
    save_figure(figure, "astar_10x10_obstacle_1_start_goal")


def render_heuristic() -> None:
    """Image 2: one node with a line to the goal labeled with distance h."""
    figure, axis = create_axes()
    draw_grid(axis)
    draw_obstacle(axis)
    draw_all_nodes(axis)
    draw_start_and_goal(axis)
    draw_heuristic_line(axis, EXAMPLE_NODE)
    highlight_node(axis, EXAMPLE_NODE, HEURISTIC_COLOR)
    annotate(axis, "h = straight-line distance to the goal (ignores the wall)")
    save_figure(figure, "astar_10x10_obstacle_2_heuristic")


def render_f_value() -> None:
    """Image 3: the example node labeled with f = g + h."""
    figure, axis = create_axes()
    draw_grid(axis)
    draw_obstacle(axis)
    draw_all_nodes(axis)
    draw_heuristic_line(axis, EXAMPLE_NODE)
    label_edge_g(axis, START_NODE, EXAMPLE_NODE)
    draw_start_and_goal(axis)
    highlight_node(axis, EXAMPLE_NODE, HEURISTIC_COLOR)

    g_value = edge_cost(START_NODE, EXAMPLE_NODE)
    h_value = heuristic(EXAMPLE_NODE)
    label_f_value(axis, EXAMPLE_NODE, g_value + h_value)
    annotate(
        axis,
        f"f = g + h = {g_value} + {h_value} = {g_value + h_value}",
    )
    save_figure(figure, "astar_10x10_obstacle_3_f_value")


def render_start_neighbors() -> None:
    """Image 4: every neighbor of the start labeled with its f = g + h."""
    figure, axis = create_axes()
    draw_grid(axis)
    draw_obstacle(axis)
    draw_all_nodes(axis)

    for neighbor in neighbors(START_NODE):
        label_edge_g(axis, START_NODE, neighbor)
    draw_start_and_goal(axis)
    for neighbor in neighbors(START_NODE):
        g_value = edge_cost(START_NODE, neighbor)
        label_f_value(axis, neighbor, g_value + heuristic(neighbor))

    annotate(axis, "Every neighbor of the start gets an f = g + h value")
    save_figure(figure, "astar_10x10_obstacle_4_start_neighbors")


def render_progress_frames() -> None:
    """Images 5+: incremental A* expansion around the wall to the goal."""
    path = find_path()
    g_values = cumulative_g(path)

    for step_index in range(len(path) - 1):
        current = path[step_index]
        committed = path[: step_index + 2]

        figure, axis = create_axes()
        draw_grid(axis)
        draw_obstacle(axis)
        draw_all_nodes(axis)

        # Score every free neighbor of the node being expanded this step.
        for neighbor in neighbors(current):
            if neighbor in path[: step_index + 1]:
                continue
            label_edge_g(axis, current, neighbor)

        draw_committed_path(axis, committed)
        highlight_node(axis, GOAL_NODE, GOAL_COLOR)
        highlight_node(axis, START_NODE, START_COLOR)

        for neighbor in neighbors(current):
            if neighbor in path[: step_index + 1]:
                continue
            g_value = g_values[step_index] + edge_cost(current, neighbor)
            label_f_value(axis, neighbor, g_value + heuristic(neighbor))

        reached_goal = committed[-1] == GOAL_NODE
        if reached_goal:
            annotate(axis, "Goal reached: the red circles trace the A* path")
        else:
            annotate(
                axis,
                "Expand the lowest-f node and add it to the path (red)",
            )
        save_figure(figure, f"astar_10x10_obstacle_5_step_{step_index + 1:02d}")


def main() -> None:
    render_start_and_goal()
    render_heuristic()
    render_f_value()
    render_start_neighbors()
    render_progress_frames()


if __name__ == "__main__":
    main()
