"""A* search around an L-shaped obstacle on an open grid, with replanning.

This is the open-space companion to the two-corridor ``astar_10x10_replan.py``.
Instead of funneling the search into a single tempting stub, it drops one
L-shaped wall onto an otherwise open grid and lets A* flood outward. The result
is a richer, more realistic picture: A* fans across the free space, and every
time the cheapest frontier node jumps to a different part of the map the search
"replans." Because the space is open, several such jumps occur -- exactly the
behavior that a narrow corridor suppresses.

Each frame visualizes the search state: the explored (closed) set, the frontier
(open) set with f = g + h labels, and the parent path from the start to the node
currently being expanded.

The L-shaped wall (rotated -90 degrees) has a horizontal arm across the middle
and a vertical arm dropping from its right end, boxing the start into the
lower-left corner so A* must round the wall. Diagonal corner-cutting is disabled
so the search never slips diagonally through the corner of the wall.
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

ORTHOGONAL_COST = 10
DIAGONAL_COST = 14
CELL_UNITS = 10  # One grid cell equals this many distance units.

# A single L-shaped wall on an otherwise open grid, rotated -90 degrees so the
# horizontal arm runs across the middle and the vertical arm drops from its right
# end. This boxes the start into the lower-left corner, so A* must round the wall
# to reach the goal. Everything else is walkable.
OBSTACLE_CELLS = frozenset(
    [(column, 4) for column in range(1, 7)]  # horizontal arm across the middle
    + [(6, row) for row in range(1, 4)]      # vertical arm drops from the right end
)

# Output files are written next to this script.
OUTPUT_DIRECTORY = Path(__file__).resolve().parent

# Shared colors.
GRID_LINE_COLOR = "#7A7A7A"
WALL_COLOR = "#252525"
NODE_FACE_COLOR = "#ECEFF1"
NODE_EDGE_COLOR = "#90A4AE"
CLOSED_COLOR = "#90CAF9"
OPEN_COLOR = "#FFE082"
CURRENT_COLOR = "#FB8C00"
START_COLOR = "#E53935"
GOAL_COLOR = "#43A047"
PATH_COLOR = "#E53935"
F_LABEL_COLOR = "#252525"


def is_free(node: tuple[int, int]) -> bool:
    """Return True if a cell is inside the map and not part of the wall."""
    column, row = node
    in_bounds = 0 <= column < GRID_SIZE and 0 <= row < GRID_SIZE
    return in_bounds and node not in OBSTACLE_CELLS


def neighbors(node: tuple[int, int]) -> list[tuple[int, int]]:
    """Return the free eight-connected neighbors of a cell.

    Diagonal moves that would clip the corner of the wall are rejected so the
    search never slips diagonally through a blocked corner.
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


def reconstruct_path(
    came_from: dict[tuple[int, int], tuple[int, int]], node: tuple[int, int]
) -> list[tuple[int, int]]:
    """Return the parent chain from the start to the given node."""
    path = [node]
    while node in came_from:
        node = came_from[node]
        path.append(node)
    path.reverse()
    return path


def run_search() -> tuple[list[dict], list[tuple[int, int]]]:
    """Run A* and return a per-expansion frame log and the final path.

    Each frame captures the state needed to draw one expansion step: the node
    being expanded, the closed and open sets, the parent path to that node, the
    f value of every open node, and a flag marking a replan (a jump to a branch
    whose parent is not the node expanded on the previous step).

    Ties in f are broken by the smaller h, so A* commits to the goal-ward node
    until the alternative is strictly cheaper.
    """
    open_heap: list[tuple[int, int, int, tuple[int, int]]] = []
    counter = 0
    heapq.heappush(
        open_heap, (heuristic(START_NODE), heuristic(START_NODE), counter, START_NODE)
    )

    open_set = {START_NODE}
    closed_set: set[tuple[int, int]] = set()
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], int] = {START_NODE: 0}
    f_score: dict[tuple[int, int], int] = {START_NODE: heuristic(START_NODE)}

    frames: list[dict] = []
    previous_node: tuple[int, int] | None = None

    while open_heap:
        _, _, _, current = heapq.heappop(open_heap)
        if current in closed_set:
            continue

        open_set.discard(current)
        closed_set.add(current)

        step_rewired: list[tuple[tuple[int, int], tuple[int, int]]] = []
        if current != GOAL_NODE:
            for neighbor in neighbors(current):
                if neighbor in closed_set:
                    continue
                tentative_g = g_score[current] + edge_cost(current, neighbor)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    if neighbor in came_from:
                        step_rewired.append((came_from[neighbor], neighbor))
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor)
                    came_from[neighbor] = current
                    counter += 1
                    heapq.heappush(
                        open_heap,
                        (f_score[neighbor], heuristic(neighbor), counter, neighbor),
                    )
                    open_set.add(neighbor)

        is_replan = (
            previous_node is not None
            and current != GOAL_NODE
            and came_from.get(current) not in (None, previous_node)
        )

        frames.append(
            {
                "current": current,
                "closed": set(closed_set),
                "open": set(open_set),
                "path": reconstruct_path(came_from, current),
                "f_open": {node: f_score[node] for node in open_set},
                "f_current": f_score[current],
                "replan": is_replan,
                "tree": dict(came_from),
                "rewired_edges": list(step_rewired),
            }
        )

        if current == GOAL_NODE:
            return frames, reconstruct_path(came_from, GOAL_NODE)

        previous_node = current

    raise ValueError("No path exists between the start and the goal.")


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


def shade_cell(axis: plt.Axes, node: tuple[int, int], color: str) -> None:
    """Fill a single cell with a translucent color to mark set membership."""
    axis.add_patch(
        Rectangle(
            (node[0] - 0.5, node[1] - 0.5),
            1,
            1,
            facecolor=color,
            edgecolor="none",
            alpha=0.85,
            zorder=1.2,
        )
    )


def draw_walls(axis: plt.Axes) -> None:
    """Fill every non-walkable cell in black."""
    for column in range(GRID_SIZE):
        for row in range(GRID_SIZE):
            if not is_free((column, row)):
                axis.add_patch(
                    Rectangle(
                        (column - 0.5, row - 0.5),
                        1,
                        1,
                        facecolor=WALL_COLOR,
                        edgecolor="none",
                        zorder=1.0,
                    )
                )


def draw_grid(axis: plt.Axes) -> None:
    """Draw the 10x10 grid lines above the shaded cells."""
    for line_index in range(GRID_SIZE + 1):
        coordinate = line_index - 0.5
        axis.plot(
            [-0.5, GRID_SIZE - 0.5],
            [coordinate, coordinate],
            color=GRID_LINE_COLOR,
            linewidth=1.2,
            zorder=1.6,
        )
        axis.plot(
            [coordinate, coordinate],
            [-0.5, GRID_SIZE - 0.5],
            color=GRID_LINE_COLOR,
            linewidth=1.2,
            zorder=1.6,
        )


def draw_free_nodes(axis: plt.Axes) -> None:
    """Draw a faint node at the center of every walkable cell."""
    free_cells = [
        (column, row)
        for column in range(GRID_SIZE)
        for row in range(GRID_SIZE)
        if is_free((column, row))
    ]
    axis.scatter(
        [column for column, _ in free_cells],
        [row for _, row in free_cells],
        s=200,
        marker="o",
        facecolor=NODE_FACE_COLOR,
        edgecolor=NODE_EDGE_COLOR,
        linewidth=1.0,
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


def draw_tree(
    axis: plt.Axes, tree: dict[tuple[int, int], tuple[int, int]]
) -> None:
    """Draw all parent-child edges in the search tree as thin gray lines."""
    for child, parent in tree.items():
        axis.plot(
            [parent[0], child[0]],
            [parent[1], child[1]],
            color="#78909C",
            linewidth=1.5,
            alpha=0.65,
            zorder=2.5,
        )


def draw_rewired_edges(
    axis: plt.Axes,
    rewired_edges: list[tuple[tuple[int, int], tuple[int, int]]],
) -> None:
    """Draw old parent connections as dashed red lines to show rewiring."""
    for old_parent, node in rewired_edges:
        axis.plot(
            [old_parent[0], node[0]],
            [old_parent[1], node[1]],
            color="#EF5350",
            linewidth=2.0,
            linestyle="--",
            alpha=0.9,
            zorder=3.5,
        )


def draw_parent_path(axis: plt.Axes, path: list[tuple[int, int]], color: str) -> None:
    """Draw the parent chain to the current node as a colored line."""
    if len(path) > 1:
        axis.plot(
            [column for column, _ in path],
            [row for _, row in path],
            color=color,
            linewidth=3.0,
            zorder=4,
        )


def label_f_value(
    axis: plt.Axes, node: tuple[int, int], f_value: int, color: str
) -> None:
    """Write a node's f value in a small box just above the node."""
    axis.text(
        node[0],
        node[1] + 0.36,
        str(f_value),
        fontsize=8.5,
        weight="bold",
        color=F_LABEL_COLOR,
        ha="center",
        va="center",
        zorder=6,
        bbox=dict(
            boxstyle="round,pad=0.14",
            facecolor="white",
            edgecolor=color,
            linewidth=1.0,
        ),
    )


def annotate(axis: plt.Axes, text: str, color: str = F_LABEL_COLOR) -> None:
    """Place an explanatory caption below the grid."""
    axis.text(
        0.5,
        -0.04,
        text,
        transform=axis.transAxes,
        fontsize=13,
        weight="bold",
        color=color,
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
def render_frame(frame: dict, file_stem: str) -> None:
    """Render one A* expansion step."""
    figure, axis = create_axes()
    draw_walls(axis)

    for node in frame["closed"]:
        shade_cell(axis, node, CLOSED_COLOR)
    for node in frame["open"]:
        shade_cell(axis, node, OPEN_COLOR)

    draw_grid(axis)
    draw_free_nodes(axis)

    draw_tree(axis, frame["tree"])
    draw_rewired_edges(axis, frame["rewired_edges"])
    draw_parent_path(axis, frame["path"], CURRENT_COLOR)
    highlight_node(axis, frame["current"], CURRENT_COLOR)
    highlight_node(axis, START_NODE, START_COLOR)
    highlight_node(axis, GOAL_NODE, GOAL_COLOR)

    for node, f_value in frame["f_open"].items():
        label_f_value(axis, node, f_value, OPEN_COLOR)
    label_f_value(axis, frame["current"], frame["f_current"], CURRENT_COLOR)

    if frame["current"] == GOAL_NODE:
        annotate(axis, "Goal reached after rounding the L-shaped wall", GOAL_COLOR)
    elif frame["replan"]:
        annotate(axis, "Frontier jumps -- A* replans to a cheaper node", "#C62828")
    else:
        annotate(axis, "Expand the lowest-f node (dive toward the goal)")

    save_figure(figure, file_stem)


def render_summary(
    path: list[tuple[int, int]], explored: set[tuple[int, int]]
) -> None:
    """Render a final frame contrasting the A* path with the explored region."""
    figure, axis = create_axes()
    draw_walls(axis)

    for node in explored:
        if node not in path:
            shade_cell(axis, node, CLOSED_COLOR)

    draw_grid(axis)
    draw_free_nodes(axis)

    draw_parent_path(axis, path, PATH_COLOR)
    for node in path:
        highlight_node(axis, node, PATH_COLOR)
    highlight_node(axis, START_NODE, START_COLOR)
    highlight_node(axis, GOAL_NODE, GOAL_COLOR)

    annotate(axis, "Final A* path (red); blue cells were explored then abandoned")
    save_figure(figure, "astar_replan_l_shape_6_summary")


def main() -> None:
    frames, path = run_search()
    for step_index, frame in enumerate(frames, start=1):
        render_frame(frame, f"astar_replan_l_shape_5_step_{step_index:02d}")
    render_summary(path, frames[-1]["closed"])


if __name__ == "__main__":
    main()
