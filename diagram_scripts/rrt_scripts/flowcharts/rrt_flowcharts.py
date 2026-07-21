"""Concise, slide-ready flowcharts of RRT and RRT*.

Two images are produced: the RRT loop (sample -> nearest -> steer -> collision
check -> add -> goal check), and RRT*, which adds the choose-best-parent and
rewire steps that make it asymptotically optimal (drawn in orange).
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon


OUTPUT_DIRECTORY = Path(__file__).resolve().parent

DARK = "#252525"
PROCESS_FILL, PROCESS_EDGE = "#E3F2FD", "#1976D2"
DECISION_FILL, DECISION_EDGE = "#FFECB3", "#F9A825"
ACCENT_FILL, ACCENT_EDGE = "#FFE0B2", "#FB8C00"
START_FILL, START_EDGE = "#BBDEFB", "#1565C0"
SUCCESS_FILL, SUCCESS_EDGE = "#C8E6C9", "#2E7D32"

SPECS = {
    "start": (3.5, 0.9, START_FILL, START_EDGE, "pill", 11),
    "success": (3.3, 0.9, SUCCESS_FILL, SUCCESS_EDGE, "pill", 11),
    "process": (4.1, 1.0, PROCESS_FILL, PROCESS_EDGE, "round", 11),
    "accent": (4.2, 1.1, ACCENT_FILL, ACCENT_EDGE, "round", 10.5),
    "decision": (3.4, 1.3, DECISION_FILL, DECISION_EDGE, "diamond", 10.5),
}


def draw_node(axis, kind, cx, cy, text):
    """Draw a node and return (cx, cy, half-height)."""
    width, height, fill, edge, shape, fontsize = SPECS[kind]
    if shape == "diamond":
        points = [(cx, cy + height / 2), (cx + width / 2, cy),
                  (cx, cy - height / 2), (cx - width / 2, cy)]
        axis.add_patch(Polygon(points, closed=True, facecolor=fill,
                               edgecolor=edge, linewidth=2, zorder=2))
    else:
        rounding = height / 2 if shape == "pill" else 0.16
        axis.add_patch(FancyBboxPatch(
            (cx - width / 2, cy - height / 2), width, height,
            boxstyle=f"round,pad=0.02,rounding_size={rounding}",
            facecolor=fill, edgecolor=edge, linewidth=2, zorder=2))
    axis.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
              color=DARK, weight="bold", zorder=3)
    return (cx, cy, height / 2)


def arrow(axis, start, end, rad=0.0, color=DARK):
    axis.annotate("", xy=end, xytext=start, zorder=1, arrowprops=dict(
        arrowstyle="-|>", color=color, linewidth=2,
        connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def connect(axis, upper, lower):
    """Vertical arrow from the bottom of one node to the top of the next."""
    arrow(axis, (upper[0], upper[1] - upper[2]), (lower[0], lower[1] + lower[2]))


def label(axis, x, y, text, color=DARK):
    axis.text(x, y, text, ha="center", va="center", fontsize=10, color=color,
              weight="bold", zorder=4,
              bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none"))


def canvas(title, xlim, ylim):
    figure, axis = plt.subplots(figsize=(7.5, (ylim[1] - ylim[0]) * 0.9))
    axis.set_title(title, fontsize=20, weight="bold", color=DARK, pad=10)
    axis.set_xlim(*xlim)
    axis.set_ylim(*ylim)
    axis.set_aspect("equal")
    axis.axis("off")
    return figure, axis


def save(figure, stem):
    png_path = OUTPUT_DIRECTORY / f"{stem}.png"
    figure.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {png_path}")


# ---------------------------------------------------------------------------
# RRT
# ---------------------------------------------------------------------------
def render_rrt():
    figure, axis = canvas("RRT", xlim=(-3.6, 6.2), ylim=(0.3, 10.3))

    start = draw_node(axis, "start", 0, 9.4, "Start: tree = {q_start}")
    sample = draw_node(axis, "process", 0, 8.1, "Sample random point  q_rand")
    nearest = draw_node(axis, "process", 0, 6.9, "Find nearest node  q_near")
    steer = draw_node(axis, "process", 0, 5.6,
                      "Steer q_near toward q_rand\nby a capped step \u2192 q_new")
    collision = draw_node(axis, "decision", 0, 4.05,
                          "edge q_near\u2013q_new\ncollision-free?")
    add = draw_node(axis, "process", 0, 2.55, "Add q_new  (parent q_near)")
    goal = draw_node(axis, "decision", 0, 1.15, "q_new reaches\nthe goal?")
    success = draw_node(axis, "success", 4.3, 1.15, "Return path")

    for upper, lower in [(start, sample), (sample, nearest), (nearest, steer),
                         (steer, collision), (collision, add), (add, goal)]:
        connect(axis, upper, lower)

    arrow(axis, (goal[0] + 3.4 / 2, goal[1]), (success[0] - 3.3 / 2, success[1]))

    # "No" loops back up to the sample step.
    arrow(axis, (collision[0] - 3.4 / 2, collision[1]),
          (sample[0] - 4.1 / 2, sample[1]), rad=-0.45, color=DECISION_EDGE)
    arrow(axis, (goal[0] - 3.4 / 2, goal[1]),
          (sample[0] - 4.1 / 2, sample[1] - 0.25), rad=-0.5, color=DECISION_EDGE)

    label(axis, 0.3, (collision[1] + add[1]) / 2, "yes")
    label(axis, -2.0, 4.05, "no", color=DECISION_EDGE)
    label(axis, (goal[0] + success[0]) / 2, 1.4, "yes")
    label(axis, -2.55, 1.15, "no", color=DECISION_EDGE)

    save(figure, "rrt_flowchart")


# ---------------------------------------------------------------------------
# RRT*
# ---------------------------------------------------------------------------
def render_rrt_star():
    figure, axis = canvas("RRT*", xlim=(-3.6, 6.2), ylim=(0.2, 11.7))

    start = draw_node(axis, "start", 0, 10.9, "Start: tree = {q_start}")
    sample = draw_node(axis, "process", 0, 9.7, "Sample random point  q_rand")
    nearest = draw_node(axis, "process", 0, 8.6, "Find nearest node  q_near")
    steer = draw_node(axis, "process", 0, 7.45,
                      "Steer q_near toward q_rand\nby a capped step \u2192 q_new")
    collision = draw_node(axis, "decision", 0, 5.95,
                          "edge q_near\u2013q_new\ncollision-free?")
    choose = draw_node(axis, "accent", 0, 4.6,
                       "Choose parent in radius\nwith lowest cost")
    add = draw_node(axis, "process", 0, 3.5, "Add q_new")
    rewire = draw_node(axis, "accent", 0, 2.35,
                       "Rewire neighbors through\nq_new if it is cheaper")
    goal = draw_node(axis, "decision", 0, 1.05, "q_new reaches\nthe goal?")
    success = draw_node(axis, "success", 4.3, 1.05, "Return path")

    for upper, lower in [(start, sample), (sample, nearest), (nearest, steer),
                         (steer, collision), (collision, choose), (choose, add),
                         (add, rewire), (rewire, goal)]:
        connect(axis, upper, lower)

    arrow(axis, (goal[0] + 3.4 / 2, goal[1]), (success[0] - 3.3 / 2, success[1]))

    arrow(axis, (collision[0] - 3.4 / 2, collision[1]),
          (sample[0] - 4.1 / 2, sample[1]), rad=-0.4, color=DECISION_EDGE)
    arrow(axis, (goal[0] - 3.4 / 2, goal[1]),
          (sample[0] - 4.1 / 2, sample[1] - 0.25), rad=-0.5, color=DECISION_EDGE)

    label(axis, 0.3, (collision[1] + choose[1]) / 2, "yes")
    label(axis, -2.0, 5.95, "no", color=DECISION_EDGE)
    label(axis, (goal[0] + success[0]) / 2, 1.3, "yes")
    label(axis, -2.55, 1.05, "no", color=DECISION_EDGE)

    save(figure, "rrt_star_flowchart")


def main():
    render_rrt()
    render_rrt_star()


if __name__ == "__main__":
    main()
