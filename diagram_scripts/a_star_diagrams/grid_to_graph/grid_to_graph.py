"""Generate a three-step figure sequence that shows how A* turns a map into a graph.

Image 1: a plain 3x3 grid drawn with black lines and a white interior.
Image 2: the same grid with a graph superimposed. Each cell holds one node
         (a circle), and edges connect every pair of neighboring nodes,
         including diagonals.
Image 3: the graph from image 2 with edge costs labeled. Orthogonal edges are
         labeled 10 and diagonal edges are labeled 14, the integer costs A*
         commonly uses to approximate 1 and sqrt(2).
"""

from pathlib import Path

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Figure settings
# ---------------------------------------------------------------------------
GRID_SIZE = 3

ORTHOGONAL_COST = 10
DIAGONAL_COST = 14

# Output files are written next to this script.
OUTPUT_DIRECTORY = Path(__file__).resolve().parent

# Shared colors so the three images look like one continuous sequence.
GRID_LINE_COLOR = "#252525"
EDGE_COLOR = "#9E9E9E"
NODE_FACE_COLOR = "#1976D2"
NODE_EDGE_COLOR = "white"
ORTHOGONAL_LABEL_COLOR = "#2E7D32"
DIAGONAL_LABEL_COLOR = "#C62828"


def node_positions(size: int = GRID_SIZE) -> list[tuple[int, int]]:
    """Return the (column, row) center of every cell in a size x size grid."""
    return [(column, row) for row in range(size) for column in range(size)]


def graph_edges(size: int = GRID_SIZE) -> list[tuple[tuple[int, int], tuple[int, int], bool]]:
    """Return unique edges between neighboring cells.

    Each edge is (node_a, node_b, is_diagonal). Every node connects to its
    eight neighbors; duplicate edges are removed by only adding an edge when
    the neighbor is ordered after the current node.
    """
    neighbor_offsets = [
        (1, 0),
        (0, 1),
        (1, 1),
        (1, -1),
    ]

    edges: list[tuple[tuple[int, int], tuple[int, int], bool]] = []
    for row in range(size):
        for column in range(size):
            for column_step, row_step in neighbor_offsets:
                neighbor_column = column + column_step
                neighbor_row = row + row_step
                if 0 <= neighbor_column < size and 0 <= neighbor_row < size:
                    is_diagonal = column_step != 0 and row_step != 0
                    edges.append(
                        ((column, row), (neighbor_column, neighbor_row), is_diagonal)
                    )
    return edges


def create_axes() -> tuple[plt.Figure, plt.Axes]:
    """Return a figure and axes prepared with a square, unlabeled canvas."""
    figure, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
    axis.set_xlim(-0.5, GRID_SIZE - 0.5)
    axis.set_ylim(-0.5, GRID_SIZE - 0.5)
    axis.set_aspect("equal")
    axis.axis("off")
    return figure, axis


def draw_grid(axis: plt.Axes, size: int = GRID_SIZE) -> None:
    """Draw a size x size grid with black cell borders and a white interior."""
    axis.set_facecolor("white")
    for line_index in range(size + 1):
        coordinate = line_index - 0.5
        axis.plot(
            [-0.5, size - 0.5],
            [coordinate, coordinate],
            color=GRID_LINE_COLOR,
            linewidth=2.5,
            zorder=1,
        )
        axis.plot(
            [coordinate, coordinate],
            [-0.5, size - 0.5],
            color=GRID_LINE_COLOR,
            linewidth=2.5,
            zorder=1,
        )


def draw_edges(axis: plt.Axes) -> None:
    """Draw edges from the center node to each of its neighbors."""
    center = (GRID_SIZE // 2, GRID_SIZE // 2)
    for node_a, node_b, _ in graph_edges():
        if center not in (node_a, node_b):
            continue
        axis.plot(
            [node_a[0], node_b[0]],
            [node_a[1], node_b[1]],
            color=EDGE_COLOR,
            linewidth=2.5,
            zorder=2,
        )


def draw_nodes(axis: plt.Axes) -> None:
    """Draw a circular node at the center of every cell."""
    columns = [column for column, _ in node_positions()]
    rows = [row for _, row in node_positions()]
    axis.scatter(
        columns,
        rows,
        s=900,
        marker="o",
        facecolor=NODE_FACE_COLOR,
        edgecolor=NODE_EDGE_COLOR,
        linewidth=2.5,
        zorder=3,
    )


def draw_edge_labels(axis: plt.Axes) -> None:
    """Label edges from the center node with A* costs: 10 for orthogonal, 14 for diagonal."""
    center = (GRID_SIZE // 2, GRID_SIZE // 2)
    for node_a, node_b, is_diagonal in graph_edges():
        if center not in (node_a, node_b):
            continue
        midpoint_column = (node_a[0] + node_b[0]) / 2
        midpoint_row = (node_a[1] + node_b[1]) / 2
        cost = DIAGONAL_COST if is_diagonal else ORTHOGONAL_COST
        label_color = DIAGONAL_LABEL_COLOR if is_diagonal else ORTHOGONAL_LABEL_COLOR
        axis.text(
            midpoint_column,
            midpoint_row,
            str(cost),
            fontsize=13,
            weight="bold",
            color=label_color,
            ha="center",
            va="center",
            zorder=4,
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="white",
                edgecolor=label_color,
                linewidth=1.2,
            ),
        )


def save_figure(figure: plt.Figure, file_stem: str) -> None:
    """Save a figure as a slide-ready PNG file, then close it."""
    png_path = OUTPUT_DIRECTORY / f"{file_stem}.png"
    figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Saved {png_path}")


def render_grid_only() -> None:
    """Image 1: the plain 3x3 grid."""
    figure, axis = create_axes()
    draw_grid(axis)
    save_figure(figure, "grid_to_graph_1_grid")


def render_grid_with_graph() -> None:
    """Image 2: the grid with nodes and edges superimposed."""
    figure, axis = create_axes()
    draw_grid(axis)
    draw_edges(axis)
    draw_nodes(axis)
    save_figure(figure, "grid_to_graph_2_graph")


def render_graph_with_costs() -> None:
    """Image 3: the graph with labeled edge costs."""
    figure, axis = create_axes()
    draw_grid(axis)
    draw_edges(axis)
    draw_edge_labels(axis)
    draw_nodes(axis)
    save_figure(figure, "grid_to_graph_3_costs")


def main() -> None:
    render_grid_only()
    render_grid_with_graph()
    render_graph_with_costs()


if __name__ == "__main__":
    main()
