#!/usr/bin/env python3
"""Illustrate trajectory-rollout local planning with a vertical hallway scene.

The world is a vertical hallway: the robot starts near the bottom, a circular
obstacle blocks the middle corridor, and the goal is near the top.

Five PNGs are exported to ``trajectory_rollout_outputs/``:

1. ``01_hallway_scene.png`` - world layout (robot, obstacle, goal, walls).
2. ``02_candidate_rollouts.png`` - dotted candidate trajectories.
3. ``03_scored_rollouts.png`` - candidates scored for viability with best path.
4. ``04_replan_after_motion.png`` - robot advances, then replans from new pose.
5. ``05_invalid_through_obstacle.png`` - highlights invalid trajectories that
    intersect the obstacle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon, Rectangle


# ---------------------------------------------------------------------------
# World configuration
# ---------------------------------------------------------------------------
WORLD_X_MIN = 0.0
WORLD_X_MAX = 4.0
WORLD_Y_MIN = 0.0
WORLD_Y_MAX = 12.0

# Corridor interior (free space lies between vertical walls).
WALL_LEFT_X = 0.3
WALL_RIGHT_X = 3.7

ROBOT_START = np.array([2.0, 1.2], dtype=float)
ROBOT_HEADING_DEG = 90.0
ROBOT_RADIUS = 0.22

GOAL_POSITION = np.array([2.0, 11.0], dtype=float)

OBSTACLE_CENTER = np.array([2.5, 5.5], dtype=float)
OBSTACLE_RADIUS = 0.75

# ---------------------------------------------------------------------------
# Rollout configuration
# ---------------------------------------------------------------------------
HORIZON_SECONDS = 2.6
TIME_STEP = 0.2
ROLLOUT_STEPS = int(round(HORIZON_SECONDS / TIME_STEP))

# Sampled command space (the "dynamic window").
LINEAR_SPEEDS = np.array([1.1, 1.6, 2.1], dtype=float)          # m/s
ANGULAR_SPEEDS = np.linspace(-0.5, 0.5, 11, dtype=float)        # rad/s

# Score weights (applied to normalized sub-scores).
WEIGHT_GOAL = 0.65
WEIGHT_CLEARANCE = 0.25
WEIGHT_SPEED = 0.10

OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "trajectory_rollout_outputs"
GIF_NAME = "06_rollout_navigation.gif"

# ---------------------------------------------------------------------------
# Visual style
# ---------------------------------------------------------------------------
BG_COLOR = "#F8FAFC"
BORDER_COLOR = "#334155"
GRID_COLOR = "#E2E8F0"
WALL_COLOR = "#475569"
WALL_HATCH_COLOR = "#94A3B8"
OBSTACLE_FACE = "#EF4444"
OBSTACLE_EDGE = "#991B1B"
ROBOT_COLOR = "#2563EB"
ROBOT_EDGE = "#1E293B"
GOAL_COLOR = "#16A34A"
CANDIDATE_COLOR = "#0EA5E9"
INVALID_COLOR = "#CBD5E1"
BEST_COLOR = "#F59E0B"
REPLAN_BEST_COLOR = "#14B8A6"
TEXT_COLOR = "#0F172A"


@dataclass
class Rollout:
    """A single forward-simulated candidate trajectory."""

    linear_speed: float
    angular_speed: float
    path: np.ndarray
    collides: bool
    min_clearance: float
    goal_progress: float
    score: float = 0.0


# ---------------------------------------------------------------------------
# Geometry and simulation helpers
# ---------------------------------------------------------------------------
def make_robot_triangle(
    position: np.ndarray,
    heading_deg: float,
    size: float = 0.42,
) -> np.ndarray:
    """Return triangle vertices for a robot glyph pointing along the heading."""
    heading = np.deg2rad(heading_deg)
    tip = position + size * np.array([np.cos(heading), np.sin(heading)])
    left = position + size * np.array(
        [np.cos(heading + 2.4), np.sin(heading + 2.4)]
    )
    right = position + size * np.array(
        [np.cos(heading - 2.4), np.sin(heading - 2.4)]
    )
    return np.array([tip, left, right])


def simulate_rollout(
    linear_speed: float,
    angular_speed: float,
    start_position: np.ndarray | None = None,
    start_heading_deg: float | None = None,
) -> np.ndarray:
    """Forward-simulate a unicycle model with a constant command."""
    if start_position is None:
        start_position = ROBOT_START
    if start_heading_deg is None:
        start_heading_deg = ROBOT_HEADING_DEG

    x, y = start_position
    theta = np.deg2rad(start_heading_deg)

    path = [np.array([x, y], dtype=float)]
    for _ in range(ROLLOUT_STEPS):
        theta += angular_speed * TIME_STEP
        x += linear_speed * np.cos(theta) * TIME_STEP
        y += linear_speed * np.sin(theta) * TIME_STEP
        path.append(np.array([x, y], dtype=float))

    return np.array(path)


def evaluate_path(path: np.ndarray) -> tuple[bool, float, float]:
    """Return collision status, min clearance, and progress toward goal."""
    collides = False
    min_clearance = np.inf

    for point in path:
        gap_to_obstacle = (
            float(np.linalg.norm(point - OBSTACLE_CENTER))
            - OBSTACLE_RADIUS
            - ROBOT_RADIUS
        )
        gap_to_walls = min(
            point[0] - (WALL_LEFT_X + ROBOT_RADIUS),
            (WALL_RIGHT_X - ROBOT_RADIUS) - point[0],
        )
        clearance = min(gap_to_obstacle, gap_to_walls)
        min_clearance = min(min_clearance, clearance)
        if clearance < 0.0:
            collides = True

    start_distance = float(np.linalg.norm(path[0] - GOAL_POSITION))
    end_distance = float(np.linalg.norm(path[-1] - GOAL_POSITION))
    goal_progress = start_distance - end_distance

    return collides, min_clearance, goal_progress


def path_intersects_obstacle(path: np.ndarray) -> bool:
    """Return True if the robot body intersects the circular obstacle."""
    for point in path:
        distance = float(np.linalg.norm(point - OBSTACLE_CENTER))
        if distance <= (OBSTACLE_RADIUS + ROBOT_RADIUS):
            return True
    return False


def generate_rollouts(
    start_position: np.ndarray | None = None,
    start_heading_deg: float | None = None,
) -> list[Rollout]:
    """Sample command space and return scored rollout candidates."""
    rollouts: list[Rollout] = []
    for linear_speed in LINEAR_SPEEDS:
        for angular_speed in ANGULAR_SPEEDS:
            path = simulate_rollout(
                linear_speed,
                angular_speed,
                start_position=start_position,
                start_heading_deg=start_heading_deg,
            )
            collides, min_clearance, goal_progress = evaluate_path(path)
            rollouts.append(
                Rollout(
                    linear_speed=linear_speed,
                    angular_speed=angular_speed,
                    path=path,
                    collides=collides,
                    min_clearance=min_clearance,
                    goal_progress=goal_progress,
                )
            )

    _assign_scores(rollouts)
    return rollouts


def _normalize(values: np.ndarray) -> np.ndarray:
    """Scale values to [0, 1], mapping flat arrays to zeros."""
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-9:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def _assign_scores(rollouts: list[Rollout]) -> None:
    """Fill composite viability score for each collision-free rollout."""
    valid = [rollout for rollout in rollouts if not rollout.collides]
    if not valid:
        return

    goal_terms = _normalize(np.array([rollout.goal_progress for rollout in valid]))
    clearance_terms = _normalize(np.array([rollout.min_clearance for rollout in valid]))
    speed_terms = _normalize(np.array([rollout.linear_speed for rollout in valid]))

    for rollout, goal_term, clearance_term, speed_term in zip(
        valid, goal_terms, clearance_terms, speed_terms
    ):
        rollout.score = (
            WEIGHT_GOAL * goal_term
            + WEIGHT_CLEARANCE * clearance_term
            + WEIGHT_SPEED * speed_term
        )


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def setup_axes(axis: plt.Axes, title: str) -> None:
    """Apply common axis styling."""
    axis.set_xlim(WORLD_X_MIN, WORLD_X_MAX)
    axis.set_ylim(WORLD_Y_MIN, WORLD_Y_MAX)
    axis.set_aspect("equal")
    axis.set_facecolor(BG_COLOR)

    for spine in axis.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color(BORDER_COLOR)

    axis.grid(True, color=GRID_COLOR, linewidth=0.7, alpha=0.8)
    axis.set_xticks(np.arange(WORLD_X_MIN, WORLD_X_MAX + 0.1, 0.5))
    axis.set_yticks(np.arange(WORLD_Y_MIN, WORLD_Y_MAX + 0.1, 1.0))
    axis.tick_params(colors=BORDER_COLOR, labelsize=8)
    axis.set_title(title, fontsize=12.8, color=TEXT_COLOR, pad=10, weight="bold")


def draw_walls(axis: plt.Axes) -> None:
    """Draw hallway side walls as hatched vertical bands."""
    left_wall = Rectangle(
        (WORLD_X_MIN, WORLD_Y_MIN),
        WALL_LEFT_X - WORLD_X_MIN,
        WORLD_Y_MAX - WORLD_Y_MIN,
        facecolor=WALL_COLOR,
        edgecolor=WALL_HATCH_COLOR,
        hatch="///",
        linewidth=0.0,
        alpha=0.9,
        zorder=1,
    )
    right_wall = Rectangle(
        (WALL_RIGHT_X, WORLD_Y_MIN),
        WORLD_X_MAX - WALL_RIGHT_X,
        WORLD_Y_MAX - WORLD_Y_MIN,
        facecolor=WALL_COLOR,
        edgecolor=WALL_HATCH_COLOR,
        hatch="///",
        linewidth=0.0,
        alpha=0.9,
        zorder=1,
    )
    axis.add_patch(left_wall)
    axis.add_patch(right_wall)


def draw_obstacle(axis: plt.Axes) -> None:
    """Draw the circular obstacle."""
    obstacle = Circle(
        OBSTACLE_CENTER,
        OBSTACLE_RADIUS,
        facecolor=OBSTACLE_FACE,
        edgecolor=OBSTACLE_EDGE,
        linewidth=1.6,
        alpha=0.92,
        zorder=6,
    )
    axis.add_patch(obstacle)
    axis.text(
        OBSTACLE_CENTER[0],
        OBSTACLE_CENTER[1],
        "obstacle",
        fontsize=8,
        color="white",
        ha="center",
        va="center",
        weight="bold",
        zorder=7,
    )


def draw_goal(axis: plt.Axes) -> None:
    """Draw the goal marker."""
    axis.scatter(
        GOAL_POSITION[0],
        GOAL_POSITION[1],
        marker="*",
        s=420,
        color=GOAL_COLOR,
        edgecolor="#14532D",
        linewidth=1.2,
        zorder=8,
    )
    axis.text(
        GOAL_POSITION[0],
        GOAL_POSITION[1] - 0.65,
        "goal",
        fontsize=9,
        color=TEXT_COLOR,
        ha="center",
        va="top",
        weight="bold",
    )


def draw_robot(
    axis: plt.Axes,
    position: np.ndarray = ROBOT_START,
    heading_deg: float = ROBOT_HEADING_DEG,
    label: str = "robot",
) -> None:
    """Draw robot body and heading glyph at any pose."""
    body = Circle(
        position,
        ROBOT_RADIUS,
        facecolor="#BFDBFE",
        edgecolor=ROBOT_EDGE,
        linewidth=1.0,
        zorder=9,
    )
    axis.add_patch(body)

    glyph = Polygon(
        make_robot_triangle(position, heading_deg),
        closed=True,
        facecolor=ROBOT_COLOR,
        edgecolor=ROBOT_EDGE,
        linewidth=1.2,
        zorder=10,
    )
    axis.add_patch(glyph)

    axis.text(
        position[0],
        position[1] - 0.45,
        label,
        fontsize=8.8,
        color=TEXT_COLOR,
        ha="center",
        va="top",
        weight="bold",
    )


def draw_base_scene(
    axis: plt.Axes,
    title: str,
    robot_position: np.ndarray = ROBOT_START,
    robot_heading_deg: float = ROBOT_HEADING_DEG,
    robot_label: str = "robot",
) -> None:
    """Render common scene elements for all figures."""
    setup_axes(axis, title)
    draw_walls(axis)
    draw_obstacle(axis)
    draw_goal(axis)
    draw_robot(axis, position=robot_position, heading_deg=robot_heading_deg, label=robot_label)


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------
def build_scene_figure() -> plt.Figure:
    """Figure 1: world layout with blocked straight-line path."""
    figure, axis = plt.subplots(figsize=(6.0, 8.8))
    draw_base_scene(axis, "Trajectory Rollout - Step 1: Vertical Hallway")

    axis.annotate(
        "",
        xy=(GOAL_POSITION[0], GOAL_POSITION[1] - 0.5),
        xytext=(ROBOT_START[0], ROBOT_START[1] + 0.45),
        arrowprops={
            "arrowstyle": "->",
            "color": "#64748B",
            "linewidth": 1.6,
            "linestyle": "--",
            "connectionstyle": "arc3,rad=0.0",
        },
        zorder=4,
    )
    axis.text(
        2.12,
        3.4,
        "direct path is blocked",
        fontsize=9,
        color="#475569",
        style="italic",
    )

    figure.tight_layout()
    return figure


def build_candidates_figure(rollouts: list[Rollout]) -> plt.Figure:
    """Figure 2: sampled dotted candidate rollouts."""
    figure, axis = plt.subplots(figsize=(6.0, 8.8))
    draw_base_scene(axis, "Trajectory Rollout - Step 2: Dotted Candidates")

    for rollout in rollouts:
        axis.plot(
            rollout.path[:, 0],
            rollout.path[:, 1],
            color=CANDIDATE_COLOR,
            linewidth=1.35,
            linestyle=":",
            alpha=0.6,
            zorder=3,
        )

    legend_handles = [
        Line2D([0], [0], color=CANDIDATE_COLOR, linewidth=1.6, linestyle=":",
               label="candidate rollout"),
    ]
    axis.legend(handles=legend_handles, loc="upper right", fontsize=9,
                framealpha=0.95)

    axis.text(
        0.015,
        0.03,
        f"{len(rollouts)} rollouts "
        f"({len(LINEAR_SPEEDS)} speeds x {len(ANGULAR_SPEEDS)} turn rates)\n"
        f"horizon {HORIZON_SECONDS:.1f} s, dt {TIME_STEP:.2f} s",
        transform=axis.transAxes,
        fontsize=8.5,
        color=TEXT_COLOR,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.92},
    )

    figure.tight_layout()
    return figure


def build_scored_figure(rollouts: list[Rollout]) -> plt.Figure:
    """Figure 3: dotted candidates scored by viability with best highlighted."""
    figure, axis = plt.subplots(figsize=(6.2, 8.8))
    draw_base_scene(axis, "Trajectory Rollout - Step 3: Score and Select")

    valid = [rollout for rollout in rollouts if not rollout.collides]
    colormap = plt.get_cmap("viridis")
    normalizer = Normalize(vmin=0.0, vmax=1.0)

    for rollout in rollouts:
        if not rollout.collides:
            continue
        axis.plot(
            rollout.path[:, 0],
            rollout.path[:, 1],
            color=INVALID_COLOR,
            linewidth=1.1,
            linestyle=":",
            alpha=0.75,
            zorder=2,
        )

    for rollout in valid:
        axis.plot(
            rollout.path[:, 0],
            rollout.path[:, 1],
            color=colormap(normalizer(rollout.score)),
            linewidth=1.8,
            linestyle=":",
            alpha=0.92,
            zorder=3,
        )

    best = max(valid, key=lambda rollout: rollout.score) if valid else None
    if best is not None:
        axis.plot(
            best.path[:, 0],
            best.path[:, 1],
            color=BEST_COLOR,
            linewidth=3.5,
            alpha=0.98,
            zorder=5,
        )

    mappable = ScalarMappable(norm=normalizer, cmap=colormap)
    mappable.set_array([])
    colorbar = figure.colorbar(mappable, ax=axis, fraction=0.025, pad=0.018)
    colorbar.set_label("viability score", fontsize=9, color=TEXT_COLOR)
    colorbar.ax.tick_params(labelsize=8, colors=BORDER_COLOR)

    legend_handles = [
        Line2D([0], [0], color=BEST_COLOR, linewidth=3.0,
               label="selected trajectory"),
        Line2D([0], [0], color=INVALID_COLOR, linewidth=1.6, linestyle=":",
               label="rejected (collision)"),
    ]
    axis.legend(handles=legend_handles, loc="upper right", fontsize=8.8,
                framealpha=0.95)

    figure.tight_layout()
    return figure


def build_replan_figure(initial_rollouts: list[Rollout]) -> plt.Figure:
    """Figure 4: move along selected path, then replan from new pose."""
    figure, axis = plt.subplots(figsize=(6.2, 8.8))

    valid_initial = [rollout for rollout in initial_rollouts if not rollout.collides]
    best_initial = max(valid_initial, key=lambda rollout: rollout.score)

    progress_index = max(2, len(best_initial.path) // 2)
    new_position = best_initial.path[progress_index]
    next_position = best_initial.path[progress_index + 1]
    direction = next_position - new_position
    new_heading_deg = float(np.rad2deg(np.arctan2(direction[1], direction[0])))

    replans = generate_rollouts(
        start_position=new_position,
        start_heading_deg=new_heading_deg,
    )
    valid_replans = [rollout for rollout in replans if not rollout.collides]
    best_replan = max(valid_replans, key=lambda rollout: rollout.score) if valid_replans else None

    draw_base_scene(
        axis,
        "Trajectory Rollout - Step 4: Move, Then Replan",
        robot_position=new_position,
        robot_heading_deg=new_heading_deg,
        robot_label="robot (replan)",
    )

    committed = best_initial.path[: progress_index + 1]
    axis.plot(
        committed[:, 0],
        committed[:, 1],
        color=BEST_COLOR,
        linewidth=3.2,
        alpha=0.98,
        zorder=5,
    )

    for rollout in replans:
        color = CANDIDATE_COLOR if not rollout.collides else INVALID_COLOR
        alpha = 0.75 if not rollout.collides else 0.65
        axis.plot(
            rollout.path[:, 0],
            rollout.path[:, 1],
            color=color,
            linewidth=1.35,
            linestyle=":",
            alpha=alpha,
            zorder=3,
        )

    if best_replan is not None:
        axis.plot(
            best_replan.path[:, 0],
            best_replan.path[:, 1],
            color=REPLAN_BEST_COLOR,
            linewidth=2.9,
            alpha=0.98,
            zorder=6,
        )

    legend_handles = [
        Line2D([0], [0], color=BEST_COLOR, linewidth=3.0,
               label="committed initial path"),
        Line2D([0], [0], color=CANDIDATE_COLOR, linewidth=1.6, linestyle=":",
               label="new candidate rollouts"),
        Line2D([0], [0], color=REPLAN_BEST_COLOR, linewidth=2.6,
               label="new selected trajectory"),
    ]
    axis.legend(handles=legend_handles, loc="upper right", fontsize=8.6,
                framealpha=0.95)

    axis.text(
        0.02,
        0.03,
        "Robot executes part of the chosen path,\nthen performs rollout scoring again.",
        transform=axis.transAxes,
        fontsize=8.4,
        color=TEXT_COLOR,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.92},
    )

    figure.tight_layout()
    return figure


def build_invalid_obstacle_figure(rollouts: list[Rollout]) -> plt.Figure:
    """Figure 5: emphasize invalid trajectories that pass through obstacle."""
    figure, axis = plt.subplots(figsize=(6.0, 8.8))
    draw_base_scene(axis, "Trajectory Rollout - Step 5: Invalid Obstacle Crossings")

    obstacle_invalid = [
        rollout for rollout in rollouts
        if rollout.collides and path_intersects_obstacle(rollout.path)
    ]
    other_invalid = [
        rollout for rollout in rollouts
        if rollout.collides and not path_intersects_obstacle(rollout.path)
    ]

    # Keep non-obstacle collisions visible but subtle for context.
    for rollout in other_invalid:
        axis.plot(
            rollout.path[:, 0],
            rollout.path[:, 1],
            color=INVALID_COLOR,
            linewidth=1.1,
            linestyle=":",
            alpha=0.45,
            zorder=2,
        )

    # Strongly highlight trajectories that pass through the obstacle.
    for rollout in obstacle_invalid:
        axis.plot(
            rollout.path[:, 0],
            rollout.path[:, 1],
            color="#DC2626",
            linewidth=2.2,
            linestyle=":",
            alpha=0.9,
            zorder=5,
        )

    legend_handles = [
        Line2D([0], [0], color="#DC2626", linewidth=2.2, linestyle=":",
               label="invalid: intersects obstacle"),
        Line2D([0], [0], color=INVALID_COLOR, linewidth=1.2, linestyle=":",
               label="other invalid trajectories"),
    ]
    axis.legend(handles=legend_handles, loc="upper right", fontsize=8.7,
                framealpha=0.95)

    axis.text(
        0.02,
        0.03,
        f"Obstacle-intersecting invalids: {len(obstacle_invalid)}\n"
        f"Other invalid trajectories: {len(other_invalid)}",
        transform=axis.transAxes,
        fontsize=8.4,
        color=TEXT_COLOR,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.92},
    )

    figure.tight_layout()
    return figure


def _heading_from_path(path: np.ndarray, index: int) -> float:
    """Estimate heading at a path index using the next segment."""
    next_index = min(index + 1, len(path) - 1)
    direction = path[next_index] - path[index]
    if float(np.linalg.norm(direction)) < 1e-9:
        return ROBOT_HEADING_DEG
    return float(np.rad2deg(np.arctan2(direction[1], direction[0])))


def _build_navigation_states() -> list[dict[str, np.ndarray | float | list[Rollout] | np.ndarray | None]]:
    """Create animation states for repeated plan-execute-replan steps."""
    states: list[dict[str, np.ndarray | float | list[Rollout] | np.ndarray | None]] = []

    current_position = ROBOT_START.copy()
    current_heading = ROBOT_HEADING_DEG
    traversed = np.array([current_position.copy()])

    max_replans = 12
    execute_steps = 3
    goal_tolerance = 0.5

    for _ in range(max_replans):
        remaining = float(np.linalg.norm(current_position - GOAL_POSITION))
        if remaining <= goal_tolerance:
            break

        rollouts = generate_rollouts(
            start_position=current_position,
            start_heading_deg=current_heading,
        )
        valid = [rollout for rollout in rollouts if not rollout.collides]
        if not valid:
            break

        best = max(valid, key=lambda rollout: rollout.score)
        states.append(
            {
                "robot_position": current_position.copy(),
                "robot_heading": current_heading,
                "rollouts": rollouts,
                "best_path": best.path,
                "traversed": traversed.copy(),
                "phase": "plan",
            }
        )

        step_count = min(execute_steps, len(best.path) - 1)
        for step in range(1, step_count + 1):
            pose = best.path[step]
            current_heading = _heading_from_path(best.path, step)
            traversed = np.vstack([traversed, pose])
            states.append(
                {
                    "robot_position": pose.copy(),
                    "robot_heading": current_heading,
                    "rollouts": rollouts,
                    "best_path": best.path,
                    "traversed": traversed.copy(),
                    "phase": "execute",
                }
            )

        current_position = best.path[step_count].copy()

    states.append(
        {
            "robot_position": current_position.copy(),
            "robot_heading": current_heading,
            "rollouts": [],
            "best_path": None,
            "traversed": traversed.copy(),
            "phase": "done",
        }
    )
    return states


def build_navigation_gif(output_path: Path) -> None:
    """Render an animated GIF of the full trajectory-rollout navigation loop."""
    states = _build_navigation_states()

    figure, axis = plt.subplots(figsize=(6.2, 8.8))

    def draw_frame(frame_index: int) -> None:
        state = states[frame_index]
        robot_position = state["robot_position"]
        robot_heading = float(state["robot_heading"])
        traversed = state["traversed"]
        phase = str(state["phase"])

        phase_label = {
            "plan": "Planning",
            "execute": "Executing",
            "done": "Finished",
        }.get(phase, "Planning")

        draw_base_scene(
            axis,
            f"Trajectory Rollout - Full Navigation ({phase_label})",
            robot_position=robot_position,
            robot_heading_deg=robot_heading,
            robot_label="robot",
        )

        if len(traversed) > 1:
            axis.plot(
                traversed[:, 0],
                traversed[:, 1],
                color=BEST_COLOR,
                linewidth=2.8,
                alpha=0.95,
                zorder=6,
            )

        rollouts_obj = state["rollouts"]
        if isinstance(rollouts_obj, list) and rollouts_obj:
            for rollout in rollouts_obj:
                if rollout.collides:
                    color = INVALID_COLOR
                    alpha = 0.35
                else:
                    color = CANDIDATE_COLOR
                    alpha = 0.55
                axis.plot(
                    rollout.path[:, 0],
                    rollout.path[:, 1],
                    color=color,
                    linewidth=1.1,
                    linestyle=":",
                    alpha=alpha,
                    zorder=3,
                )

        best_path_obj = state["best_path"]
        if isinstance(best_path_obj, np.ndarray):
            axis.plot(
                best_path_obj[:, 0],
                best_path_obj[:, 1],
                color=REPLAN_BEST_COLOR,
                linewidth=2.2,
                alpha=0.92,
                zorder=5,
            )

        distance_to_goal = float(np.linalg.norm(robot_position - GOAL_POSITION))
        axis.text(
            0.02,
            0.03,
            f"Frame: {frame_index + 1}/{len(states)}\n"
            f"Distance to goal: {distance_to_goal:.2f}",
            transform=axis.transAxes,
            fontsize=8.5,
            color=TEXT_COLOR,
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.92},
        )

    animation = FuncAnimation(
        figure,
        draw_frame,
        frames=len(states),
        interval=260,
        repeat=True,
    )
    animation.save(output_path, writer=PillowWriter(fps=4))
    plt.close(figure)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Generate and save all trajectory-rollout diagrams."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    rollouts = generate_rollouts()

    figures = {
        "01_hallway_scene.png": build_scene_figure(),
        "02_candidate_rollouts.png": build_candidates_figure(rollouts),
        "03_scored_rollouts.png": build_scored_figure(rollouts),
        "04_replan_after_motion.png": build_replan_figure(rollouts),
        "05_invalid_through_obstacle.png": build_invalid_obstacle_figure(rollouts),
    }

    for filename, figure in figures.items():
        output_path = OUTPUT_DIRECTORY / filename
        figure.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight")
        plt.close(figure)
        print(f"Saved {output_path}")

    gif_path = OUTPUT_DIRECTORY / GIF_NAME
    build_navigation_gif(gif_path)
    print(f"Saved {gif_path}")

    valid_count = sum(1 for rollout in rollouts if not rollout.collides)
    print(
        f"Evaluated {len(rollouts)} rollouts "
        f"({valid_count} collision-free, {len(rollouts) - valid_count} rejected)."
    )


if __name__ == "__main__":
    main()
