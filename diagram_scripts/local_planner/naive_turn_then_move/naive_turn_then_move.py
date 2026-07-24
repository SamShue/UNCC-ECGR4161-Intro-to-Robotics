#!/usr/bin/env python3
"""Create a naive local-planner diagram sequence and animated GIF.

Behavior model:
1) Rotate in place until heading aligns with the active goal.
2) Drive straight to that goal.
3) Repeat for the next goal.

The script plans through three goals in order and exports:
- PNG keyframes only (initial, turn-complete, goal-reached)
- Animated GIF composed from the same steps
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WORLD_MIN = 0.0
WORLD_MAX = 10.0

START_POSITION = np.array([1.2, 1.2], dtype=float)
START_HEADING_DEG = 35.0

GOALS = [
    np.array([8.2, 2.0], dtype=float),
    np.array([8.0, 8.3], dtype=float),
    np.array([2.0, 7.8], dtype=float),
]

TURN_RATE_DEG_PER_STEP = 8.0
MOVE_STEP = 0.22
HEADING_TOLERANCE_DEG = 1.2
GOAL_TOLERANCE = 0.05

FRAME_PREFIX = "naive_turn_then_move"
GIF_NAME = "naive_turn_then_move.gif"
FRAME_RATE = 12

# Visual style
BG_COLOR = "#F8FAFC"
BORDER_COLOR = "#334155"
GRID_COLOR = "#CBD5E1"
PATH_COLOR = "#0F766E"
TURN_RAY_COLOR = "#EA580C"
MOVE_RAY_COLOR = "#0EA5E9"
ROBOT_COLOR = "#2563EB"
ROBOT_EDGE = "#1E293B"
GOAL_PENDING = "#94A3B8"
GOAL_ACTIVE = "#F59E0B"
GOAL_DONE = "#16A34A"
TEXT_COLOR = "#0F172A"


@dataclass
class FrameState:
    """Single renderable state of the naive planner."""

    position: np.ndarray
    heading_deg: float
    goal_index: int
    phase: str
    reached_count: int
    trajectory: list[np.ndarray]


def wrap_angle_deg(angle_deg: float) -> float:
    """Normalize an angle to [-180, 180)."""
    return (angle_deg + 180.0) % 360.0 - 180.0


def angle_to_goal_deg(position: np.ndarray, goal: np.ndarray) -> float:
    """Return world-frame heading from position to goal in degrees."""
    delta = goal - position
    return float(np.degrees(np.arctan2(delta[1], delta[0])))


def make_robot_triangle(position: np.ndarray, heading_deg: float) -> np.ndarray:
    """Create a small triangle robot footprint in world coordinates."""
    heading = np.deg2rad(heading_deg)
    forward = np.array([np.cos(heading), np.sin(heading)])
    left = np.array([-forward[1], forward[0]])

    length = 0.35
    width = 0.23

    tip = position + length * forward
    rear_left = position - 0.55 * length * forward + width * left
    rear_right = position - 0.55 * length * forward - width * left
    return np.vstack([tip, rear_left, rear_right])


def generate_plan_states() -> list[FrameState]:
    """Simulate a turn-then-move policy across all goals."""
    states: list[FrameState] = []

    position = START_POSITION.copy()
    heading = START_HEADING_DEG
    reached_count = 0
    trajectory: list[np.ndarray] = [position.copy()]

    states.append(
        FrameState(
            position=position.copy(),
            heading_deg=heading,
            goal_index=0,
            phase="initial",
            reached_count=reached_count,
            trajectory=[node.copy() for node in trajectory],
        )
    )

    for goal_index, goal in enumerate(GOALS):
        target_heading = angle_to_goal_deg(position, goal)

        while True:
            error = wrap_angle_deg(target_heading - heading)
            if abs(error) <= HEADING_TOLERANCE_DEG:
                heading = target_heading
                states.append(
                    FrameState(
                        position=position.copy(),
                        heading_deg=heading,
                        goal_index=goal_index,
                        phase="turn",
                        reached_count=reached_count,
                        trajectory=[node.copy() for node in trajectory],
                    )
                )
                break

            step = np.sign(error) * min(TURN_RATE_DEG_PER_STEP, abs(error))
            heading = wrap_angle_deg(heading + step)

            states.append(
                FrameState(
                    position=position.copy(),
                    heading_deg=heading,
                    goal_index=goal_index,
                    phase="turn",
                    reached_count=reached_count,
                    trajectory=[node.copy() for node in trajectory],
                )
            )

        while True:
            delta = goal - position
            distance = float(np.linalg.norm(delta))
            if distance <= GOAL_TOLERANCE:
                position = goal.copy()
                trajectory.append(position.copy())
                reached_count += 1
                states.append(
                    FrameState(
                        position=position.copy(),
                        heading_deg=heading,
                        goal_index=goal_index,
                        phase="arrive",
                        reached_count=reached_count,
                        trajectory=[node.copy() for node in trajectory],
                    )
                )
                break

            step_dist = min(MOVE_STEP, distance)
            move_dir = np.array(
                [np.cos(np.deg2rad(heading)), np.sin(np.deg2rad(heading))],
                dtype=float,
            )
            position = position + step_dist * move_dir
            trajectory.append(position.copy())

            states.append(
                FrameState(
                    position=position.copy(),
                    heading_deg=heading,
                    goal_index=goal_index,
                    phase="move",
                    reached_count=reached_count,
                    trajectory=[node.copy() for node in trajectory],
                )
            )

    return states


def draw_scene(axis: plt.Axes, state: FrameState) -> None:
    """Render one planner state onto the provided axis."""
    axis.clear()
    axis.set_xlim(WORLD_MIN, WORLD_MAX)
    axis.set_ylim(WORLD_MIN, WORLD_MAX)
    axis.set_aspect("equal")
    axis.set_facecolor(BG_COLOR)

    for spine in axis.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color(BORDER_COLOR)

    axis.grid(True, color=GRID_COLOR, linewidth=0.7, alpha=0.8)
    axis.set_xticks(np.arange(WORLD_MIN, WORLD_MAX + 0.1, 1.0))
    axis.set_yticks(np.arange(WORLD_MIN, WORLD_MAX + 0.1, 1.0))
    axis.tick_params(colors=BORDER_COLOR, labelsize=8)

    path_points = np.array(state.trajectory)
    axis.plot(path_points[:, 0], path_points[:, 1], color=PATH_COLOR, linewidth=2.2, alpha=0.95)

    for idx, goal in enumerate(GOALS):
        if idx < state.reached_count:
            color = GOAL_DONE
        elif idx == state.goal_index:
            color = GOAL_ACTIVE
        else:
            color = GOAL_PENDING

        axis.scatter(goal[0], goal[1], s=110, color=color, edgecolor="#111827", linewidth=0.7, zorder=5)
        axis.text(
            goal[0] + 0.15,
            goal[1] + 0.15,
            f"G{idx + 1}",
            fontsize=9,
            color=TEXT_COLOR,
            weight="bold",
        )

    active_goal = GOALS[state.goal_index]
    ray_color = TURN_RAY_COLOR if state.phase == "turn" else MOVE_RAY_COLOR
    axis.plot(
        [state.position[0], active_goal[0]],
        [state.position[1], active_goal[1]],
        linestyle="--",
        linewidth=1.5,
        color=ray_color,
        alpha=0.9,
        zorder=3,
    )

    robot_poly = Polygon(
        make_robot_triangle(state.position, state.heading_deg),
        closed=True,
        facecolor=ROBOT_COLOR,
        edgecolor=ROBOT_EDGE,
        linewidth=1.2,
        zorder=10,
    )
    axis.add_patch(robot_poly)

    phase_label = {
        "initial": "Initial Pose",
        "turn": "Turning Toward Goal",
        "move": "Moving Straight",
        "arrive": "Reached Goal",
    }[state.phase]

    axis.set_title(
        f"Naive Local Planner: Goal {state.goal_index + 1}/3 | {phase_label}",
        fontsize=12,
        color=TEXT_COLOR,
        pad=10,
    )

    axis.text(
        0.02,
        0.02,
        f"Heading: {state.heading_deg:6.1f} deg\n"
        f"Position: ({state.position[0]:.2f}, {state.position[1]:.2f})\n"
        f"Completed goals: {state.reached_count}/3",
        transform=axis.transAxes,
        fontsize=9,
        color=TEXT_COLOR,
        bbox={"facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.92},
    )


def select_keyframe_states(states: list[FrameState]) -> list[FrameState]:
    """Return key states: initial, each turn completion, and each arrival."""
    if not states:
        return []

    keyframes: list[FrameState] = [states[0]]
    for i in range(1, len(states)):
        previous = states[i - 1]
        current = states[i]

        turn_completed = previous.phase == "turn" and current.phase == "move"
        if turn_completed or current.phase == "arrive":
            keyframes.append(previous if turn_completed else current)

    return keyframes


def save_keyframe_sequence(states: list[FrameState], output_dir: Path) -> None:
    """Write PNGs for key planner states only."""
    for stale_frame in output_dir.glob(f"{FRAME_PREFIX}_*.png"):
        stale_frame.unlink()

    keyframes = select_keyframe_states(states)

    figure, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)

    for i, state in enumerate(keyframes):
        draw_scene(axis, state)
        frame_path = output_dir / f"{FRAME_PREFIX}_{i:03d}.png"
        figure.savefig(frame_path, dpi=140)

    plt.close(figure)


def save_gif(states: list[FrameState], gif_path: Path) -> None:
    """Create an animated GIF from the planner states."""
    figure, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)

    def update(frame_index: int):
        draw_scene(axis, states[frame_index])
        return []

    animation = FuncAnimation(
        figure,
        update,
        frames=len(states),
        interval=1000 / FRAME_RATE,
        blit=False,
        repeat=True,
    )

    writer = PillowWriter(fps=FRAME_RATE)
    animation.save(str(gif_path), writer=writer, dpi=120)
    plt.close(figure)


def main() -> None:
    """Run simulation and export diagrams."""
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "naive_turn_then_move_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    states = generate_plan_states()
    if not states:
        raise RuntimeError("No states were generated.")

    save_keyframe_sequence(states, output_dir)
    save_gif(states, output_dir / GIF_NAME)

    keyframe_count = len(select_keyframe_states(states))
    print(f"Saved {keyframe_count} keyframe PNGs to: {output_dir}")
    print(f"Saved GIF to: {output_dir / GIF_NAME}")


if __name__ == "__main__":
    main()
