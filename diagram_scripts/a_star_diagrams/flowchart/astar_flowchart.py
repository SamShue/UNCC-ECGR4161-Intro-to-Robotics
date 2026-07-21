"""Draw a concise flowchart of the A* path planner used in these diagrams.

The chart is sized to sit on a single slide: a start terminal, the OPEN/goal
decisions, the expand-and-relax step, and the two exits (no path / reconstruct
path), with a loop back to the top.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon


OUTPUT_DIRECTORY = Path(__file__).resolve().parent

DARK = "#252525"
PROCESS_FILL, PROCESS_EDGE = "#E3F2FD", "#1976D2"
DECISION_FILL, DECISION_EDGE = "#FFECB3", "#F9A825"
RELAX_FILL, RELAX_EDGE = "#FFE0B2", "#FB8C00"
START_FILL, START_EDGE = "#BBDEFB", "#1565C0"
SUCCESS_FILL, SUCCESS_EDGE = "#C8E6C9", "#2E7D32"
FAIL_FILL, FAIL_EDGE = "#FFCDD2", "#C62828"


def box(axis, cx, cy, w, h, text, fill, edge, rounding=0.16, fontsize=12):
    """Draw a rounded rectangle with centered text."""
    axis.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        facecolor=fill, edgecolor=edge, linewidth=2, zorder=2,
    ))
    axis.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
              color=DARK, weight="bold", zorder=3)


def terminal(axis, cx, cy, w, h, text, fill, edge, fontsize=12):
    """Draw a pill-shaped terminal node."""
    box(axis, cx, cy, w, h, text, fill, edge, rounding=h / 2, fontsize=fontsize)


def diamond(axis, cx, cy, w, h, text, fill, edge, fontsize=12):
    """Draw a decision diamond with centered text."""
    points = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    axis.add_patch(Polygon(points, closed=True, facecolor=fill, edgecolor=edge,
                           linewidth=2, zorder=2))
    axis.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
              color=DARK, weight="bold", zorder=3)


def arrow(axis, start, end, rad=0.0, color=DARK):
    """Draw an arrow between two points."""
    axis.annotate("", xy=end, xytext=start, zorder=1, arrowprops=dict(
        arrowstyle="-|>", color=color, linewidth=2,
        connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0,
    ))


def label(axis, x, y, text, color=DARK):
    """Place a small edge label (Yes / No)."""
    axis.text(x, y, text, ha="center", va="center", fontsize=10.5,
              color=color, weight="bold", zorder=4,
              bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none"))


def main() -> None:
    figure, axis = plt.subplots(figsize=(8.5, 9.2))

    # Node centers.
    start = (0, 9.3)
    d_open = (0, 7.8)
    pop = (0, 6.35)
    d_goal = (0, 4.95)
    closed = (0, 3.65)
    relax = (0, 1.95)
    fail = (4.1, 7.8)
    success = (4.1, 4.95)

    proc_w, proc_h = 3.7, 0.95
    dia_w, dia_h = 3.0, 1.25
    term_w, term_h = 3.1, 0.95
    relax_w, relax_h = 4.1, 1.55

    terminal(axis, *start, term_w, term_h, "Add start to OPEN\n(g = 0,   f = h)",
             START_FILL, START_EDGE)
    diamond(axis, *d_open, dia_w, dia_h, "OPEN\nempty?", DECISION_FILL, DECISION_EDGE)
    box(axis, *pop, proc_w, proc_h, "Pop lowest-f\nnode from OPEN", PROCESS_FILL, PROCESS_EDGE)
    diamond(axis, *d_goal, dia_w, dia_h, "Node =\ngoal?", DECISION_FILL, DECISION_EDGE)
    box(axis, *closed, proc_w, proc_h, "Move node to CLOSED", PROCESS_FILL, PROCESS_EDGE)
    box(axis, *relax, relax_w, relax_h,
        "For each free neighbor:\nif  g + step cost < g(neighbor):\n"
        "   update g,   f = g + h,   parent\n   add neighbor to OPEN",
        RELAX_FILL, RELAX_EDGE, fontsize=10.5)
    terminal(axis, *fail, term_w, term_h, "No path found", FAIL_FILL, FAIL_EDGE)
    terminal(axis, *success, term_w, term_h, "Reconstruct path", SUCCESS_FILL, SUCCESS_EDGE)

    # Main downward flow.
    arrow(axis, (0, start[1] - term_h / 2), (0, d_open[1] + dia_h / 2))
    arrow(axis, (0, d_open[1] - dia_h / 2), (0, pop[1] + proc_h / 2))
    arrow(axis, (0, pop[1] - proc_h / 2), (0, d_goal[1] + dia_h / 2))
    arrow(axis, (0, d_goal[1] - dia_h / 2), (0, closed[1] + proc_h / 2))
    arrow(axis, (0, closed[1] - proc_h / 2), (0, relax[1] + relax_h / 2))

    # Decision branches to the exits.
    arrow(axis, (d_open[0] + dia_w / 2, d_open[1]), (fail[0] - term_w / 2, fail[1]))
    arrow(axis, (d_goal[0] + dia_w / 2, d_goal[1]), (success[0] - term_w / 2, success[1]))

    # Loop back from the relax step to the OPEN-empty test.
    arrow(axis, (relax[0] - relax_w / 2, relax[1]),
          (d_open[0] - dia_w / 2, d_open[1]), rad=-0.42, color=PROCESS_EDGE)

    # Yes / No labels.
    label(axis, 0.28, (d_open[1] + pop[1]) / 2, "No")
    label(axis, (d_open[0] + dia_w / 2 + fail[0] - term_w / 2) / 2, d_open[1] + 0.25, "Yes")
    label(axis, 0.28, (d_goal[1] + closed[1]) / 2, "No")
    label(axis, (d_goal[0] + dia_w / 2 + success[0] - term_w / 2) / 2, d_goal[1] + 0.25, "Yes")
    label(axis, -2.55, (relax[1] + d_open[1]) / 2, "loop", color=PROCESS_EDGE)

    axis.set_title("A* Path Planning", fontsize=20, weight="bold", color=DARK, pad=12)
    axis.set_xlim(-3.2, 6.0)
    axis.set_ylim(0.7, 10.2)
    axis.set_aspect("equal")
    axis.axis("off")

    png_path = OUTPUT_DIRECTORY / "astar_flowchart.png"
    figure.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
