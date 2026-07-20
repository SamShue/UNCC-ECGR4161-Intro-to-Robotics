"""Show how a cell's cost is relaxed to a lower value while A* builds its tree.

This is the grid companion to the other A* diagrams. It runs A* on a small open
grid with one short wall and focuses on the "relaxation" step: when a cell is
reached from a cheaper neighbor, its cost-so-far g is lowered and the branch of
the search tree that reaches it is rewired to the cheaper parent. The tree
therefore always keeps, for every cell, the single branch with its lowest cost.

Movement is eight-connected with the usual costs (10 orthogonal, 14 diagonal)
and the same rounded straight-line heuristic as the other scripts. Because that
heuristic is not exact around the wall, A* sometimes reaches a cell by an
expensive branch first and then relaxes it -- exactly the moment this sequence
highlights.

Each frame shows the grid and wall, the explored (closed) and frontier (open)
cells, each discovered cell's cost as f = g + h (cost-so-far plus the distance-
to-goal estimate), the search tree as arrows from each parent to its child, and
-- on a relaxation -- the cost as "old -> new", the new branch in orange, and the
discarded branch as a faded dashed arrow. A relaxation lowers g (and therefore
f) while h, the distance to the goal, stays fixed.
"""

import heapq
from math import hypot
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ---------------------------------------------------------------------------
# Map settings
# ---------------------------------------------------------------------------
GRID_SIZE = 5

# Coordinates use the form (column, row), with (0, 0) at the lower-left.
START_NODE = (0, 0)
GOAL_NODE = (4, 4)

# After this node is expanded, insert a frame that computes the cost to its left
# and below neighbors and shows why they are not updated (the cost is higher).
REJECT_DEMO_NODE = (1, 1)

ORTHOGONAL_COST = 10
DIAGONAL_COST = 14
CELL_UNITS = 10  # One grid cell equals this many distance units.

# A short wall that makes the heuristic imperfect, so A* reaches some cells by an
# expensive branch first and later relaxes them to a cheaper one.
OBSTACLE_CELLS = frozenset([(2, 0), (2, 1), (2, 2)])

# Output files are written next to this script.
OUTPUT_DIRECTORY = Path(__file__).resolve().parent

# Shared colors (matching the other A* diagrams).
GRID_LINE_COLOR = "#B8B8B8"
WALL_COLOR = "#252525"
NODE_FACE_COLOR = "#ECEFF1"
NODE_EDGE_COLOR = "#90A4AE"
CLOSED_COLOR = "#90CAF9"
OPEN_COLOR = "#FFE082"
CURRENT_COLOR = "#FB8C00"
START_COLOR = "#E53935"
GOAL_COLOR = "#43A047"
TREE_COLOR = "#2E7D32"
RELAX_COLOR = "#FB8C00"
REJECT_COLOR = "#C62828"
DARK = "#252525"


def is_free(node: tuple[int, int]) -> bool:
    """Return True if a cell is inside the map and not part of the wall."""
    column, row = node
    in_bounds = 0 <= column < GRID_SIZE and 0 <= row < GRID_SIZE
    return in_bounds and node not in OBSTACLE_CELLS


def neighbors(node: tuple[int, int]) -> list[tuple[int, int]]:
    """Return the free eight-connected neighbors of a cell (no corner cutting)."""
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


def run_search() -> list[dict]:
    """Run A* and return a per-expansion log that captures every relaxation."""
    open_heap: list[tuple[int, int, int, tuple[int, int]]] = []
    counter = 0
    heapq.heappush(
        open_heap, (heuristic(START_NODE), heuristic(START_NODE), counter, START_NODE)
    )

    open_set = {START_NODE}
    closed_set: set[tuple[int, int]] = set()
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], int] = {START_NODE: 0}

    frames = [
        {
            "extracted": None,
            "g": dict(g_score),
            "parent": dict(parent),
            "closed": set(),
            "open": set(open_set),
            "relaxed": [],
            "discovered": [],
            "rejected": [],
            "caption": "Initialize: cost(start) = 0; expand the lowest-f cell each step",
        }
    ]

    while open_heap:
        _, _, _, current = heapq.heappop(open_heap)
        if current in closed_set:
            continue
        open_set.discard(current)
        closed_set.add(current)

        relaxed: list[tuple[tuple[int, int], int, int, tuple[int, int] | None]] = []
        discovered: list[tuple[int, int]] = []
        rejected: list[tuple[tuple[int, int], int, int]] = []

        if current != GOAL_NODE:
            for neighbor in neighbors(current):
                if neighbor in closed_set:
                    continue
                candidate = g_score[current] + edge_cost(current, neighbor)
                if neighbor not in g_score:
                    discovered.append(neighbor)
                elif candidate < g_score[neighbor]:
                    relaxed.append((neighbor, g_score[neighbor], candidate, parent.get(neighbor)))
                else:
                    rejected.append((neighbor, candidate, g_score[neighbor]))
                    continue
                g_score[neighbor] = candidate
                parent[neighbor] = current
                counter += 1
                heapq.heappush(
                    open_heap,
                    (candidate + heuristic(neighbor), heuristic(neighbor), counter, neighbor),
                )
                open_set.add(neighbor)

        frames.append(
            {
                "extracted": current,
                "g": dict(g_score),
                "parent": dict(parent),
                "closed": set(closed_set),
                "open": set(open_set),
                "relaxed": relaxed,
                "discovered": discovered,
                "rejected": rejected,
                "caption": build_caption(current, relaxed, discovered),
            }
        )

        if current == GOAL_NODE:
            break

    return frames


def build_caption(
    node: tuple[int, int],
    relaxed: list[tuple[tuple[int, int], int, int, tuple[int, int] | None]],
    discovered: list[tuple[int, int]],
) -> str:
    """Return a short description of one expansion step."""
    if node == GOAL_NODE:
        return f"Expand goal {node}: f = g (h = 0), costs are final"
    if relaxed:
        cell, old_g, new_g, _ = relaxed[0]
        h = heuristic(cell)
        extra = f" (+{len(relaxed) - 1} more)" if len(relaxed) > 1 else ""
        return (
            f"Expand {node}: f{cell} lowered {old_g + h}→{new_g + h} "
            f"(g {old_g}→{new_g}, h {h} fixed){extra}"
        )
    if discovered:
        return f"Expand {node}: discover {len(discovered)} neighbor(s), set f = g + h"
    return f"Expand {node}"


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def create_axes() -> tuple[plt.Figure, plt.Axes]:
    """Return a figure and axes prepared with a square, unlabeled canvas."""
    figure, axis = plt.subplots(figsize=(7.5, 7.5), constrained_layout=True)
    axis.set_xlim(-0.5, GRID_SIZE - 0.5)
    axis.set_ylim(-0.5, GRID_SIZE - 0.5)
    axis.set_aspect("equal")
    axis.axis("off")
    return figure, axis


def shade_cell(axis: plt.Axes, node: tuple[int, int], color: str) -> None:
    """Fill a single cell with a translucent color to mark set membership."""
    axis.add_patch(
        Rectangle(
            (node[0] - 0.5, node[1] - 0.5), 1, 1,
            facecolor=color, edgecolor="none", alpha=0.85, zorder=1.2,
        )
    )


def draw_walls(axis: plt.Axes) -> None:
    """Fill every wall cell in black."""
    for column in range(GRID_SIZE):
        for row in range(GRID_SIZE):
            if not is_free((column, row)):
                axis.add_patch(
                    Rectangle(
                        (column - 0.5, row - 0.5), 1, 1,
                        facecolor=WALL_COLOR, edgecolor="none", zorder=1.0,
                    )
                )


def draw_grid(axis: plt.Axes) -> None:
    """Draw the grid lines above the shaded cells."""
    for line_index in range(GRID_SIZE + 1):
        coordinate = line_index - 0.5
        axis.plot([-0.5, GRID_SIZE - 0.5], [coordinate, coordinate],
                  color=GRID_LINE_COLOR, linewidth=1.0, zorder=1.6)
        axis.plot([coordinate, coordinate], [-0.5, GRID_SIZE - 0.5],
                  color=GRID_LINE_COLOR, linewidth=1.0, zorder=1.6)


def draw_free_nodes(axis: plt.Axes) -> None:
    """Draw a faint node at the center of every free cell."""
    free_cells = [
        (column, row)
        for column in range(GRID_SIZE)
        for row in range(GRID_SIZE)
        if is_free((column, row))
    ]
    axis.scatter(
        [c for c, _ in free_cells], [r for _, r in free_cells],
        s=180, marker="o", facecolor=NODE_FACE_COLOR,
        edgecolor=NODE_EDGE_COLOR, linewidth=1.0, zorder=2,
    )


def highlight_node(axis: plt.Axes, node: tuple[int, int], color: str, size: int = 300) -> None:
    """Draw a single filled circle over a cell to emphasize it."""
    axis.scatter(node[0], node[1], s=size, marker="o", facecolor=color,
                 edgecolor="white", linewidth=2.0, zorder=5)


def draw_tree_arrow(axis: plt.Axes, parent: tuple[int, int], child: tuple[int, int],
                    color: str, dashed: bool) -> None:
    """Draw an arrow from a parent cell to a child cell along the tree."""
    axis.annotate(
        "", xy=child, xytext=parent, zorder=3,
        arrowprops=dict(
            arrowstyle="-|>", color=color,
            linewidth=2.6 if not dashed else 1.8,
            linestyle="--" if dashed else "-",
            shrinkA=11, shrinkB=11, alpha=0.5 if dashed else 1.0,
        ),
    )


def label_cost(axis: plt.Axes, node: tuple[int, int], text: str, border: str,
               fontsize: float = 8) -> None:
    """Write a cost label in a small box just above a cell (f on top, g+h below)."""
    axis.text(
        node[0], node[1] + 0.36, text, fontsize=fontsize, weight="bold", color=DARK,
        ha="center", va="center", zorder=6, linespacing=1.05,
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor=border, linewidth=1.1),
    )


def annotate(axis: plt.Axes, text: str, color: str = DARK) -> None:
    """Place an explanatory caption below the grid."""
    axis.text(0.5, -0.04, text, transform=axis.transAxes, fontsize=12,
              weight="bold", color=color, ha="center", va="top")


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
    """Render one relaxation / tree-growth step."""
    figure, axis = create_axes()
    draw_walls(axis)

    for node in frame["closed"]:
        shade_cell(axis, node, CLOSED_COLOR)
    for node in frame["open"]:
        shade_cell(axis, node, OPEN_COLOR)

    draw_grid(axis)
    draw_free_nodes(axis)

    relaxed_nodes = {item[0]: item for item in frame["relaxed"]}
    discovered = set(frame["discovered"])

    # Faded dashed arrows for branches discarded by a relaxation this step.
    for cell, _old, _new, old_parent in frame["relaxed"]:
        if old_parent is not None:
            draw_tree_arrow(axis, old_parent, cell, "#90A4AE", dashed=True)

    # Current tree: an arrow from each cell's parent to the cell.
    for node, parent in frame["parent"].items():
        color = RELAX_COLOR if node in relaxed_nodes or node in discovered else TREE_COLOR
        draw_tree_arrow(axis, parent, node, color, dashed=False)

    # Rejected-cost computations: a red dashed candidate arrow that is not kept.
    for cell, _candidate, _current in frame.get("reject_demo", []):
        draw_tree_arrow(axis, frame["extracted"], cell, REJECT_COLOR, dashed=True)

    # Start and goal markers.
    highlight_node(axis, START_NODE, START_COLOR)
    highlight_node(axis, GOAL_NODE, GOAL_COLOR)
    if frame["extracted"] is not None:
        axis.scatter(frame["extracted"][0], frame["extracted"][1], s=360, marker="o",
                     facecolor="none", edgecolor=DARK, linewidth=2.5, zorder=6)

    # Cost labels above every discovered cell: f on top, g + h below.
    for node, value in frame["g"].items():
        h = heuristic(node)
        if node in relaxed_nodes:
            _, old_g, new_g, _ = relaxed_nodes[node]
            text = f"{old_g + h}\u2192{new_g + h}\ng {old_g}\u2192{new_g}"
            label_cost(axis, node, text, RELAX_COLOR, fontsize=7.5)
        else:
            border = RELAX_COLOR if node in discovered else DARK
            label_cost(axis, node, f"{value + h}\n{value}+{h}", border)

    # Rejection notes: the higher candidate cost, labeled on the dashed arrow
    # inside the grid so this frame matches the layout of the other steps.
    for cell, candidate, current in frame.get("reject_demo", []):
        expanded = frame["extracted"]
        mid_x = (expanded[0] + cell[0]) / 2
        mid_y = (expanded[1] + cell[1]) / 2
        axis.text(
            mid_x, mid_y, f"{candidate} \u2717",
            fontsize=8.5, weight="bold", color=REJECT_COLOR, ha="center", va="center",
            zorder=7,
            bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                      edgecolor=REJECT_COLOR, linewidth=1.1),
        )

    if frame.get("reject_demo"):
        caption_color = REJECT_COLOR
    elif frame["relaxed"]:
        caption_color = RELAX_COLOR
    else:
        caption_color = DARK
    annotate(axis, frame["caption"], caption_color)
    save_figure(figure, file_stem)


def render_summary(final: dict) -> None:
    """Render the completed shortest-path tree with final costs."""
    figure, axis = create_axes()
    draw_walls(axis)
    for node in final["closed"]:
        shade_cell(axis, node, CLOSED_COLOR)
    draw_grid(axis)
    draw_free_nodes(axis)

    for node, parent in final["parent"].items():
        draw_tree_arrow(axis, parent, node, TREE_COLOR, dashed=False)

    highlight_node(axis, START_NODE, START_COLOR)
    highlight_node(axis, GOAL_NODE, GOAL_COLOR)
    for node, value in final["g"].items():
        h = heuristic(node)
        label_cost(axis, node, f"{value + h}\n{value}+{h}", TREE_COLOR)

    annotate(axis, "Search tree: every cell keeps the branch with its lowest cost", TREE_COLOR)
    save_figure(figure, "cost_relaxation_summary")


def insert_reject_demo(frames: list[dict]) -> None:
    """Insert a frame after REJECT_DEMO_NODE's expansion showing higher-cost,
    not-updated computations for its left and below neighbors."""
    node = REJECT_DEMO_NODE
    left = (node[0] - 1, node[1])
    below = (node[0], node[1] - 1)
    for index, frame in enumerate(frames):
        if frame["extracted"] != node:
            continue
        entries = [item for item in frame["rejected"] if item[0] in (left, below)]
        if not entries:
            return
        candidate = entries[0][1]
        current = entries[0][2]
        demo = dict(frame)
        demo["relaxed"] = []
        demo["discovered"] = []
        demo["reject_demo"] = entries
        demo["caption"] = (
            f"Expand {node}: reaching {left} / {below} costs {candidate} "
            f"\u2265 {current} -- no update"
        )
        frames.insert(index + 1, demo)
        return


def main() -> None:
    frames = run_search()
    insert_reject_demo(frames)
    for step_index, frame in enumerate(frames):
        render_frame(frame, f"cost_relaxation_step_{step_index:02d}")
    render_summary(frames[-1])


if __name__ == "__main__":
    main()
