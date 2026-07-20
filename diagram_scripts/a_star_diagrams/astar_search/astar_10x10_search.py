"""Generate an A* teaching sequence on a 10x10 obstacle-free map.

The images build up the idea of A* cost computation, f = g + h, where:
    g = accumulated edge cost from the start (10 orthogonal, 14 diagonal)
    h = straight-line (Euclidean) distance from a node to the goal, scaled so
        that one cell equals 10 units and rounded to an integer
    f = g + h = the value A* uses to choose which node to expand next

Image sequence:
    1. The 10x10 grid with the graph nodes, a red start node, and a green goal.
    2. One node with a line drawn to the goal, labeled with its distance h.
    3. That same node labeled with f = g + h, showing the edge cost g and the
       heuristic h that combine into the node's value.
    4. Every neighbor of the start labeled with its own f = g + h value.
    5+. Incremental A* expansion frames that march toward the goal. Nodes that
        have been committed to the path are highlighted as red circles.
"""

from math import hypot
from pathlib import Path

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Map settings
# ---------------------------------------------------------------------------
GRID_SIZE = 10

# Coordinates use the form (column, row), with (0, 0) at the lower-left.
START_NODE = (1, 1)
GOAL_NODE = (8, 8)

# The example node used to introduce h and f in images 2 and 3.
EXAMPLE_NODE = (2, 2)

ORTHOGONAL_COST = 10
DIAGONAL_COST = 14
CELL_UNITS = 10  # One grid cell equals this many distance units.

# Output files are written next to this script.
OUTPUT_DIRECTORY = Path(__file__).resolve().parent

# Shared colors so the whole sequence looks continuous.
GRID_LINE_COLOR = "#252525"
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


def sign(value: int) -> int:
    """Return -1, 0, or 1 matching the sign of value."""
    return (value > 0) - (value < 0)


def neighbors(node: tuple[int, int]) -> list[tuple[int, int]]:
    """Return the in-bounds eight-connected neighbors of a cell."""
    column, row = node
    result = []
    for column_step in (-1, 0, 1):
        for row_step in (-1, 0, 1):
            if column_step == 0 and row_step == 0:
                continue
            neighbor = (column + column_step, row + row_step)
            if 0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE:
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


def optimal_path() -> list[tuple[int, int]]:
    """Return the obstacle-free least-cost path from start to goal.

    With no obstacles the cheapest route steps diagonally toward the goal
    until aligned, then straight, which each move greedily reproduces.
    """
    path = [START_NODE]
    column, row = START_NODE
    while (column, row) != GOAL_NODE:
        column += sign(GOAL_NODE[0] - column)
        row += sign(GOAL_NODE[1] - row)
        path.append((column, row))
    return path


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


def draw_all_nodes(axis: plt.Axes) -> None:
    """Draw a faint node at the center of every cell."""
    columns = [column for column in range(GRID_SIZE) for _ in range(GRID_SIZE)]
    rows = [row for _ in range(GRID_SIZE) for row in range(GRID_SIZE)]
    axis.scatter(
        columns,
        rows,
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
    """Image 1: the grid with the start (red) and goal (green) nodes."""
    figure, axis = create_axes()
    draw_grid(axis)
    draw_all_nodes(axis)
    draw_start_and_goal(axis)
    save_figure(figure, "astar_10x10_1_start_goal")


def render_heuristic() -> None:
    """Image 2: one node with a line to the goal labeled with distance h."""
    figure, axis = create_axes()
    draw_grid(axis)
    draw_all_nodes(axis)
    draw_start_and_goal(axis)
    draw_heuristic_line(axis, EXAMPLE_NODE)
    highlight_node(axis, EXAMPLE_NODE, HEURISTIC_COLOR)
    annotate(axis, "h = straight-line distance from the node to the goal")
    save_figure(figure, "astar_10x10_2_heuristic")


def render_f_value() -> None:
    """Image 3: the example node labeled with f = g + h."""
    figure, axis = create_axes()
    draw_grid(axis)
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
    save_figure(figure, "astar_10x10_3_f_value")


def render_start_neighbors() -> None:
    """Image 4: every neighbor of the start labeled with its f = g + h."""
    figure, axis = create_axes()
    draw_grid(axis)
    draw_all_nodes(axis)

    for neighbor in neighbors(START_NODE):
        label_edge_g(axis, START_NODE, neighbor)
    draw_start_and_goal(axis)
    for neighbor in neighbors(START_NODE):
        g_value = edge_cost(START_NODE, neighbor)
        label_f_value(axis, neighbor, g_value + heuristic(neighbor))

    annotate(axis, "Every neighbor of the start gets an f = g + h value")
    save_figure(figure, "astar_10x10_4_start_neighbors")


def render_progress_frames() -> None:
    """Images 5+: incremental A* expansion until the goal is reached."""
    path = optimal_path()
    g_values = cumulative_g(path)

    for step_index in range(len(path) - 1):
        current = path[step_index]
        committed = path[: step_index + 2]

        figure, axis = create_axes()
        draw_grid(axis)
        draw_all_nodes(axis)

        # Score every neighbor of the node being expanded this step.
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
        save_figure(figure, f"astar_10x10_5_step_{step_index + 1:02d}")


def main() -> None:
    render_start_and_goal()
    render_heuristic()
    render_f_value()
    render_start_neighbors()
    render_progress_frames()


if __name__ == "__main__":
    main()
