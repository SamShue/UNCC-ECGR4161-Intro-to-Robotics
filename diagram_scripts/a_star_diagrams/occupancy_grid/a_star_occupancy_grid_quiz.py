"""Generate quiz-ready A* occupancy-grid PNGs with g/h labels.

The script builds a small occupancy grid with two L-shaped obstacles, runs A*
on the map, then renders a question image showing an in-progress search state.
The frontier is annotated with g and h values only; the combined f value is not
displayed in the image. Four candidate frontier cells are labeled A, B, C, and
D so the quiz can ask which cell A* should expand next.

Two PNGs are written next to this script:
    * a question image with the labeled frontier cells
    * an answer image that highlights the correct next cell
"""

import heapq
from math import hypot
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ---------------------------------------------------------------------------
# Map settings
# ---------------------------------------------------------------------------
GRID_SIZE = 8

# Coordinates use the form (column, row), with (0, 0) at the lower-left.
START_NODE = (1, 1)
GOAL_NODE = (6, 6)

ORTHOGONAL_COST = 10
DIAGONAL_COST = 14
CELL_UNITS = 10

# Two L-shaped obstacles in the grid.
OBSTACLE_CELLS = frozenset(
    {
        # Lower L-shape.
        (2, 2), (3, 2), (4, 2), (2, 3), (2, 4),
        # Upper L-shape.
        (3, 5), (4, 5), (5, 5), (5, 4), (5, 3),
    }
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
G_LABEL_COLOR = "#00695C"
H_LABEL_COLOR = "#1976D2"
OPTION_LABEL_COLOR = "#1F1F1F"


def is_free(node: tuple[int, int]) -> bool:
    """Return True if a cell is inside the map and not part of a wall."""
    column, row = node
    in_bounds = 0 <= column < GRID_SIZE and 0 <= row < GRID_SIZE
    return in_bounds and node not in OBSTACLE_CELLS


def neighbors(node: tuple[int, int]) -> list[tuple[int, int]]:
    """Return the free eight-connected neighbors of a cell."""
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


def run_search() -> list[dict]:
    """Run A* and return a frame log for each node expansion."""
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

    while open_heap:
        _, _, _, current = heapq.heappop(open_heap)
        if current in closed_set:
            continue

        open_set.discard(current)
        closed_set.add(current)

        if current != GOAL_NODE:
            for neighbor in neighbors(current):
                if neighbor in closed_set:
                    continue
                tentative_g = g_score[current] + edge_cost(current, neighbor)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor)
                    came_from[neighbor] = current
                    counter += 1
                    heapq.heappush(
                        open_heap,
                        (f_score[neighbor], heuristic(neighbor), counter, neighbor),
                    )
                    open_set.add(neighbor)

        frames.append(
            {
                "current": current,
                "closed": set(closed_set),
                "open": set(open_set),
                "path": reconstruct_path(came_from, current),
                "tree": dict(came_from),
                "g": dict(g_score),
                "f": dict(f_score),
            }
        )

        if current == GOAL_NODE:
            break

    return frames


def select_quiz_frame(frames: list[dict]) -> dict:
    """Choose a later frame where the search has explored several cells."""
    candidate_frames = [
        frame
        for frame in frames
        if frame["current"] != START_NODE and len(frame["closed"]) >= 5 and 4 <= len(frame["open"]) <= 10
    ]
    if candidate_frames:
        return candidate_frames[0]

    candidate_frames = [frame for frame in frames if len(frame["open"]) >= 4]
    if candidate_frames:
        return candidate_frames[min(2, len(candidate_frames) - 1)]

    return frames[-1]


def choose_options(frame: dict) -> list[tuple[int, int]]:
    """Pick four frontier nodes to label as A, B, C, and D."""
    ranked_open_nodes = sorted(
        frame["open"],
        key=lambda node: (frame["f"][node], node[1], node[0]),
    )
    return ranked_open_nodes[:4]


def option_labels(
    frame: dict, options: list[tuple[int, int]]
) -> tuple[dict[tuple[int, int], str], str]:
    """Assign visible labels to the options and return the correct label."""
    displayed_order = sorted(options, key=lambda node: (node[1], node[0]))
    label_names = ["A", "B", "C", "D"]
    label_map = {node: label for node, label in zip(displayed_order, label_names)}

    correct_node = min(options, key=lambda node: frame["f"][node])
    return label_map, label_map[correct_node]


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


def shade_cell(axis: plt.Axes, node: tuple[int, int], color: str, alpha: float = 0.85) -> None:
    """Fill a single cell with a translucent color."""
    axis.add_patch(
        Rectangle(
            (node[0] - 0.5, node[1] - 0.5),
            1,
            1,
            facecolor=color,
            edgecolor="none",
            alpha=alpha,
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
    """Draw one grid line around each occupancy-grid cell."""
    for line_index in range(GRID_SIZE + 1):
        coordinate = line_index - 0.5
        axis.plot(
            [-0.5, GRID_SIZE - 0.5],
            [coordinate, coordinate],
            color=GRID_LINE_COLOR,
            linewidth=1.1,
            zorder=1.6,
        )
        axis.plot(
            [coordinate, coordinate],
            [-0.5, GRID_SIZE - 0.5],
            color=GRID_LINE_COLOR,
            linewidth=1.1,
            zorder=1.6,
        )


def draw_free_nodes(axis: plt.Axes) -> None:
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
        s=190,
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


def draw_parent_path(axis: plt.Axes, path: list[tuple[int, int]]) -> None:
    """Draw the parent chain to the current node as a red line."""
    if len(path) > 1:
        axis.plot(
            [column for column, _ in path],
            [row for _, row in path],
            color=PATH_COLOR,
            linewidth=3.0,
            zorder=4,
        )


def draw_tree(axis: plt.Axes, tree: dict[tuple[int, int], tuple[int, int]]) -> None:
    """Draw the discovered parent-child connections as a faint tree."""
    for child, parent in tree.items():
        axis.plot(
            [parent[0], child[0]],
            [parent[1], child[1]],
            color="#78909C",
            linewidth=1.5,
            alpha=0.65,
            zorder=2.6,
        )


def label_cell_values(
    axis: plt.Axes,
    node: tuple[int, int],
    g_value: int,
    h_value: int,
    edge_color: str,
) -> None:
    """Write g and h in the center of a cell without showing f."""
    axis.text(
        node[0],
        node[1] + 0.02,
        f"g = {g_value}\nh = {h_value}",
        fontsize=8.8,
        weight="bold",
        color="#1F1F1F",
        ha="center",
        va="center",
        zorder=6,
        bbox=dict(
            boxstyle="round,pad=0.18",
            facecolor="white",
            edgecolor=edge_color,
            linewidth=1.0,
        ),
    )


def label_option(axis: plt.Axes, node: tuple[int, int], label: str) -> None:
    """Place a quiz option label just above a frontier cell."""
    axis.text(
        node[0],
        node[1] + 0.38,
        label,
        fontsize=12,
        weight="bold",
        color=OPTION_LABEL_COLOR,
        ha="center",
        va="center",
        zorder=7,
        bbox=dict(
            boxstyle="round,pad=0.15",
            facecolor="white",
            edgecolor=OPTION_LABEL_COLOR,
            linewidth=1.0,
        ),
    )


def annotate(axis: plt.Axes, text: str, color: str = "#1F1F1F") -> None:
    """Place a caption below the grid."""
    axis.text(
        0.5,
        -0.05,
        text,
        transform=axis.transAxes,
        fontsize=13,
        weight="bold",
        color=color,
        ha="center",
        va="top",
    )


def save_figure(figure: plt.Figure, file_stem: str) -> None:
    """Save a figure as a PNG file, then close it."""
    png_path = OUTPUT_DIRECTORY / f"{file_stem}.png"
    figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Saved {png_path}")


# ---------------------------------------------------------------------------
# Image renderers
# ---------------------------------------------------------------------------
def render_question(frame: dict, option_map: dict[tuple[int, int], str]) -> None:
    """Render the quiz question image."""
    figure, axis = create_axes()
    draw_walls(axis)

    for node in frame["closed"]:
        if node != START_NODE and node != GOAL_NODE:
            shade_cell(axis, node, CLOSED_COLOR)
    for node in frame["open"]:
        shade_cell(axis, node, OPEN_COLOR)

    draw_grid(axis)
    draw_free_nodes(axis)
    draw_tree(axis, frame["tree"])

    for node in frame["closed"] | frame["open"]:
        if node in frame["g"]:
            edge_color = CLOSED_COLOR if node in frame["closed"] else OPEN_COLOR
            label_cell_values(axis, node, frame["g"][node], heuristic(node), edge_color)

    highlight_node(axis, START_NODE, START_COLOR)
    highlight_node(axis, GOAL_NODE, GOAL_COLOR)
    highlight_node(axis, frame["current"], CURRENT_COLOR)

    for node, label in option_map.items():
        label_option(axis, node, label)

    annotate(axis, "Which cell should A* explore next?")
    save_figure(figure, "a_star_occupancy_grid_quiz_question")


def render_answer(frame: dict, option_map: dict[tuple[int, int], str], correct_label: str) -> None:
    """Render the answer key image with the correct option highlighted."""
    figure, axis = create_axes()
    draw_walls(axis)

    for node in frame["closed"]:
        if node != START_NODE and node != GOAL_NODE:
            shade_cell(axis, node, CLOSED_COLOR)
    for node in frame["open"]:
        shade_cell(axis, node, OPEN_COLOR)

    draw_grid(axis)
    draw_free_nodes(axis)
    draw_tree(axis, frame["tree"])

    for node in frame["closed"] | frame["open"]:
        if node in frame["g"]:
            edge_color = CLOSED_COLOR if node in frame["closed"] else OPEN_COLOR
            label_cell_values(axis, node, frame["g"][node], heuristic(node), edge_color)

    highlight_node(axis, START_NODE, START_COLOR)
    highlight_node(axis, GOAL_NODE, GOAL_COLOR)
    highlight_node(axis, frame["current"], CURRENT_COLOR)

    for node, label in option_map.items():
        if label == correct_label:
            highlight_node(axis, node, GOAL_COLOR, size=360)
        label_option(axis, node, label)

    annotate(axis, f"Correct next cell: {correct_label}", GOAL_COLOR)
    save_figure(figure, "a_star_occupancy_grid_quiz_answer")


def main() -> None:
    """Build the quiz images."""
    frames = run_search()
    quiz_frame = select_quiz_frame(frames)
    options = choose_options(quiz_frame)
    option_map, correct_label = option_labels(quiz_frame, options)

    render_question(quiz_frame, option_map)
    render_answer(quiz_frame, option_map, correct_label)


if __name__ == "__main__":
    main()