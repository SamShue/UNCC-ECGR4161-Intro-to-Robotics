"""Build the intuition behind RRT with an incremental sampling sequence.

The frames start from an empty space with only a start and a goal, then grow a
branch one random step at a time until it reaches the goal. It is a deliberately
simple, single-branch version of the "sample a point, step toward it" idea that
Rapidly-exploring Random Trees (RRT) generalize.

Frames:
    sampling_step_0 -- empty space with the start and goal.
    sampling_step_1..4 -- each adds one random displacement, the last reaching
    the goal.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
BOUNDS = 10.0
START = np.array([1.5, 1.5])
GOAL = np.array([8.7, 8.7])

OUTPUT_DIRECTORY = Path(__file__).resolve().parent

BORDER_COLOR = "#607D8B"
NODE_COLOR = "#1976D2"
EDGE_COLOR = "#1976D2"
NEW_COLOR = "#FB8C00"
START_COLOR = "#43A047"
GOAL_COLOR = "#E53935"


def build_nodes() -> list[np.ndarray]:
    """Return the chain of nodes: start, three random steps, then the goal."""
    rng = np.random.default_rng(7)
    fractions = [0.28, 0.53, 0.77]
    nodes = [START]
    for fraction in fractions:
        base = START + fraction * (GOAL - START)
        jitter = rng.uniform(-1.1, 1.1, size=2)
        nodes.append(np.clip(base + jitter, 0.6, BOUNDS - 0.6))
    nodes.append(GOAL)
    return nodes


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def create_axes() -> tuple[plt.Figure, plt.Axes]:
    """Return a figure and axes framing the environment."""
    figure, axis = plt.subplots(figsize=(7, 7.4))
    axis.add_patch(plt.Rectangle((0, 0), BOUNDS, BOUNDS, facecolor="#FAFAFA",
                                 edgecolor=BORDER_COLOR, linewidth=2.0, zorder=0))
    axis.set_xlim(-0.4, BOUNDS + 0.4)
    axis.set_ylim(-0.4, BOUNDS + 0.4)
    axis.set_aspect("equal")
    axis.axis("off")
    return figure, axis


def render_frame(step: int, nodes: list[np.ndarray]) -> None:
    """Render the tree after `step` random displacements."""
    figure, axis = create_axes()

    # Committed edges; the newest one is drawn as an orange displacement arrow.
    for index in range(step):
        a, b = nodes[index], nodes[index + 1]
        if index == step - 1:
            axis.annotate("", xy=b, xytext=a, zorder=4, arrowprops=dict(
                arrowstyle="-|>", color=NEW_COLOR, linewidth=3.0))
        else:
            axis.plot([a[0], b[0]], [a[1], b[1]], color=EDGE_COLOR,
                      linewidth=2.5, zorder=3)

    # Tree nodes (the newest one highlighted).
    for index in range(1, step + 1):
        color = NEW_COLOR if index == step else NODE_COLOR
        axis.scatter(*nodes[index], s=150, color=color, edgecolor="white",
                     linewidth=1.8, zorder=5)

    # Start and goal always on top.
    axis.scatter(*START, s=340, marker="o", facecolor=START_COLOR,
                 edgecolor="white", linewidth=2.0, zorder=6)
    axis.scatter(*GOAL, s=430, marker="*", facecolor=GOAL_COLOR,
                 edgecolor="white", linewidth=1.8, zorder=6)
    axis.text(START[0], START[1] - 0.7, "start", ha="center", va="top",
              fontsize=12, weight="bold", color=START_COLOR, zorder=6)
    axis.text(GOAL[0], GOAL[1] + 0.7, "goal", ha="center", va="bottom",
              fontsize=12, weight="bold", color=GOAL_COLOR, zorder=6)

    png_path = OUTPUT_DIRECTORY / f"sampling_step_{step}.png"
    figure.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {png_path}")


def main() -> None:
    nodes = build_nodes()
    for step in range(len(nodes)):  # 0 (blank) through 4 (reaches goal)
        render_frame(step, nodes)


if __name__ == "__main__":
    main()
