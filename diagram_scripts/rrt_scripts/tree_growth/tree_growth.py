"""Show a Rapidly-exploring Random Tree developing branch by branch.

This bridges the single-branch "sampling idea" to a full RRT: in an empty space,
the tree repeatedly samples a random point, finds its nearest node, and extends a
short step toward it. Across the frames the tree fans out from the root and fills
the space, and the final frame highlights the branch that reaches the goal.

Each frame shows more of the same tree; nodes and edges added since the previous
frame are drawn in orange so the growth between slides is obvious.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
BOUNDS = 10.0
START = np.array([2.0, 2.0])
GOAL = np.array([8.5, 8.5])

STEP_SIZE = 1.2
GOAL_BIAS = 0.06
GOAL_THRESHOLD = 1.2
MAX_NODES = 120
SEED = 3

OUTPUT_DIRECTORY = Path(__file__).resolve().parent

BORDER_COLOR = "#607D8B"
EDGE_COLOR = "#64B5F6"
NODE_COLOR = "#1E88E5"
NEW_COLOR = "#FB8C00"
PATH_COLOR = "#C62828"
START_COLOR = "#43A047"
GOAL_COLOR = "#E53935"


def grow_tree():
    """Grow an RRT in open space; return nodes, parents, and the goal index.

    The first few nodes are placed by hand with tiny steps to show the tree
    branching: one node off the start, another off that node, then a second
    branch back off the start. Growth then continues with the normal RRT step.
    """
    rng = np.random.default_rng(SEED)
    nodes = [START]
    parents = [-1]

    # Tiny explicit first steps: (position, parent index).
    for position, parent in [
        (np.array([2.6, 2.2]), 0),   # node 1 connects back to the start
        (np.array([3.1, 2.7]), 1),   # node 2 connects back to node 1
        (np.array([1.7, 2.6]), 0),   # node 3 starts a second branch off the start
    ]:
        nodes.append(position)
        parents.append(parent)

    goal_index = None

    while len(nodes) < MAX_NODES:
        sample = GOAL if rng.random() < GOAL_BIAS else rng.uniform(0.6, BOUNDS - 0.6, 2)
        node_array = np.array(nodes)
        distances = np.hypot(node_array[:, 0] - sample[0], node_array[:, 1] - sample[1])
        nearest_index = int(distances.argmin())
        nearest = nodes[nearest_index]

        direction = sample - nearest
        length = float(np.hypot(*direction))
        if length < 1e-9:
            continue
        new_node = nearest + direction / length * min(STEP_SIZE, length)

        nodes.append(new_node)
        parents.append(nearest_index)

        if goal_index is None and np.hypot(*(new_node - GOAL)) <= GOAL_THRESHOLD:
            nodes.append(GOAL.copy())
            parents.append(len(nodes) - 2)
            goal_index = len(nodes) - 1

    return nodes, parents, goal_index


def reconstruct_path(nodes, parents, goal_index):
    """Return coordinates along the path from the goal back to the root."""
    path = []
    index = goal_index
    while index != -1:
        path.append(nodes[index])
        index = parents[index]
    return np.array(path[::-1])


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def create_axes() -> tuple[plt.Figure, plt.Axes]:
    """Return a figure and axes framing the environment."""
    figure, axis = plt.subplots(figsize=(7, 7))
    axis.add_patch(plt.Rectangle((0, 0), BOUNDS, BOUNDS, facecolor="#FAFAFA",
                                 edgecolor=BORDER_COLOR, linewidth=2.0, zorder=0))
    axis.set_xlim(-0.4, BOUNDS + 0.4)
    axis.set_ylim(-0.4, BOUNDS + 0.4)
    axis.set_aspect("equal")
    axis.axis("off")
    return figure, axis


def render_frame(index, count, previous, nodes, parents, goal_index, draw_path):
    """Render the tree up to `count` nodes; new pieces (>= previous) in orange."""
    figure, axis = create_axes()

    for node_index in range(1, count):
        a, b = nodes[parents[node_index]], nodes[node_index]
        is_new = node_index >= previous
        axis.plot([a[0], b[0]], [a[1], b[1]],
                  color=NEW_COLOR if is_new else EDGE_COLOR,
                  linewidth=2.0 if is_new else 1.3,
                  zorder=4 if is_new else 3, solid_capstyle="round")

    for node_index in range(1, count):
        is_new = node_index >= previous
        axis.scatter(*nodes[node_index], s=45 if is_new else 26,
                     color=NEW_COLOR if is_new else NODE_COLOR,
                     edgecolor="white", linewidth=0.8, zorder=5)

    if draw_path and goal_index is not None and goal_index < count:
        path = reconstruct_path(nodes, parents, goal_index)
        axis.plot(path[:, 0], path[:, 1], color=PATH_COLOR, linewidth=3.5, zorder=6)

    axis.scatter(*START, s=340, marker="o", facecolor=START_COLOR,
                 edgecolor="white", linewidth=2.0, zorder=7)
    axis.scatter(*GOAL, s=430, marker="*", facecolor=GOAL_COLOR,
                 edgecolor="white", linewidth=1.8, zorder=7)

    png_path = OUTPUT_DIRECTORY / f"tree_step_{index}.png"
    figure.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {png_path}")


def main() -> None:
    nodes, parents, goal_index = grow_tree()
    total = len(nodes)
    # Add one node at a time first (to show branching), then jump ahead.
    counts = sorted({1, 2, 3, 4, max(6, round(total * 0.5)), total})

    previous = 1
    for index, count in enumerate(counts):
        is_last = count == counts[-1]
        # Suppress the orange highlight on the final frame so the path reads clearly.
        highlight_from = count if is_last else previous
        render_frame(index, count, highlight_from, nodes, parents, goal_index,
                     draw_path=is_last)
        previous = count


if __name__ == "__main__":
    main()
