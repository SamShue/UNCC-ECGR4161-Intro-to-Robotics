"""Illustrate how a tree node rewires to a cheaper parent, as RRT* does.

RRT* does not just grow a tree -- it reshapes it. When a new node offers an
existing node a cheaper route back to the root, that node rewires its parent to
the new one, and its whole subtree comes along. This four-frame sequence shows a
node reached by a long, winding branch being rewired to a much shorter path.

Frames:
    rewiring_step_0 -- the node X sits at the end of a long, high-cost branch.
    rewiring_step_1 -- a new node N is added close to X.
    rewiring_step_2 -- through N, X would have a cheaper route (candidate edge).
    rewiring_step_3 -- X rewires its parent to N; the tree reshapes.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Tree definition
# ---------------------------------------------------------------------------
POSITIONS = {
    "R": (1.5, 1.5),
    "A": (2.7, 3.4),
    "B": (0.9, 4.0),
    "C": (2.0, 6.3),
    "D": (3.6, 7.4),
    "X": (5.0, 5.9),
    "Y": (6.6, 6.3),
    "N": (4.3, 4.0),
}
# Parents before any rewiring: a short stub R->A and a long branch to X.
BASE_PARENTS = {"A": "R", "B": "R", "C": "B", "D": "C", "X": "D", "Y": "X"}

OUTPUT_DIRECTORY = Path(__file__).resolve().parent

BORDER_COLOR = "#607D8B"
NODE_COLOR = "#1976D2"
EDGE_COLOR = "#1976D2"
NEW_COLOR = "#FB8C00"
REWIRE_COLOR = "#2E7D32"
REMOVED_COLOR = "#B0BEC5"
ROOT_COLOR = "#43A047"
HIGH_COST_COLOR = "#C62828"
DARK = "#252525"

CAPTIONS = [
    "X is reached by a long, winding branch (high cost)",
    "A new node N is added close to X",
    "Through N, X has a cheaper route to the root",
    "X rewires its parent to N \u2014 the tree reshapes",
]


def path_cost(node: str, parents: dict[str, str]) -> float:
    """Return the total edge length from the root to a node."""
    total = 0.0
    current = node
    while current in parents:
        parent = parents[current]
        total += float(np.hypot(*(np.subtract(POSITIONS[current], POSITIONS[parent]))))
        current = parent
    return total


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def create_axes() -> tuple[plt.Figure, plt.Axes]:
    """Return a figure and axes framing the environment."""
    figure, axis = plt.subplots(figsize=(7, 7.4))
    axis.add_patch(plt.Rectangle((0.2, 0.7), 7.1, 7.6, facecolor="#FAFAFA",
                                 edgecolor=BORDER_COLOR, linewidth=2.0, zorder=0))
    axis.set_xlim(-0.1, 7.6)
    axis.set_ylim(0.4, 8.6)
    axis.set_aspect("equal")
    axis.axis("off")
    return figure, axis


def draw_edge(axis, parent, child, color, width=2.5, dashed=False, alpha=1.0, zorder=3):
    """Draw a tree edge between two named nodes."""
    (xa, ya), (xb, yb) = POSITIONS[parent], POSITIONS[child]
    axis.plot([xa, xb], [ya, yb], color=color, linewidth=width,
              linestyle="--" if dashed else "-", alpha=alpha,
              solid_capstyle="round", zorder=zorder)


def cost_label(axis, node, text, color):
    """Write a cost label in a box next to a node."""
    x, y = POSITIONS[node]
    axis.text(x + 0.55, y, text, ha="left", va="center", fontsize=12,
              weight="bold", color=color, zorder=8,
              bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                        edgecolor=color, linewidth=1.4))


def render_frame(step: int) -> None:
    """Render one frame of the rewiring sequence."""
    figure, axis = create_axes()

    parents = dict(BASE_PARENTS)
    if step >= 1:
        parents["N"] = "A"
    if step == 3:
        parents["X"] = "N"

    highlight = {}
    if step in (1, 2):
        highlight[("A", "N")] = NEW_COLOR
    if step == 3:
        highlight[("N", "X")] = REWIRE_COLOR

    # The discarded edge lingers, faded, on the rewire frame.
    if step == 3:
        draw_edge(axis, "D", "X", REMOVED_COLOR, width=2.0, dashed=True, alpha=0.9, zorder=2)

    # Tree edges from the current parent map.
    for child, parent in parents.items():
        color = highlight.get((parent, child), EDGE_COLOR)
        width = 3.6 if (parent, child) in highlight else 2.5
        draw_edge(axis, parent, child, color, width=width)

    # Candidate cheaper edge under consideration.
    if step == 2:
        draw_edge(axis, "N", "X", REWIRE_COLOR, width=2.5, dashed=True, zorder=4)

    # Nodes.
    for name, (x, y) in POSITIONS.items():
        if name == "N" and step == 0:
            continue
        if name == "R":
            axis.scatter(x, y, s=360, marker="o", facecolor=ROOT_COLOR,
                         edgecolor="white", linewidth=2.0, zorder=6)
        elif name == "N" and step in (1, 2):
            axis.scatter(x, y, s=190, marker="o", facecolor=NEW_COLOR,
                         edgecolor="white", linewidth=1.8, zorder=6)
        elif name == "X":
            face = REWIRE_COLOR if step == 3 else NODE_COLOR
            axis.scatter(x, y, s=230, marker="o", facecolor=face,
                         edgecolor=DARK, linewidth=2.0, zorder=6)
        else:
            axis.scatter(x, y, s=150, marker="o", facecolor=NODE_COLOR,
                         edgecolor="white", linewidth=1.6, zorder=6)

    axis.text(*POSITIONS["R"], "  root", ha="left", va="center", fontsize=11,
              weight="bold", color=ROOT_COLOR, zorder=7)
    axis.text(POSITIONS["X"][0], POSITIONS["X"][1] - 0.5, "X", ha="center",
              va="top", fontsize=12, weight="bold", color=DARK, zorder=7)
    if step >= 1:
        axis.text(POSITIONS["N"][0], POSITIONS["N"][1] - 0.5, "N", ha="center",
                  va="top", fontsize=12, weight="bold", color=DARK, zorder=7)

    # Cost callouts on X.
    old_cost = path_cost("X", BASE_PARENTS)
    rewired_parents = {**BASE_PARENTS, "N": "A", "X": "N"}
    new_cost = path_cost("X", rewired_parents)
    if step in (0, 1):
        cost_label(axis, "X", f"cost \u2248 {old_cost:.1f}", HIGH_COST_COLOR)
    elif step == 2:
        cost_label(axis, "X", f"{old_cost:.1f} \u2192 {new_cost:.1f}?", REWIRE_COLOR)
    else:
        cost_label(axis, "X", f"cost \u2248 {new_cost:.1f}", REWIRE_COLOR)

    axis.text(0.5, -0.03, CAPTIONS[step], transform=axis.transAxes,
              fontsize=13, weight="bold", color=DARK, ha="center", va="top")

    png_path = OUTPUT_DIRECTORY / f"rewiring_step_{step}.png"
    figure.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {png_path}")


def main() -> None:
    for step in range(4):
        render_frame(step)


if __name__ == "__main__":
    main()
