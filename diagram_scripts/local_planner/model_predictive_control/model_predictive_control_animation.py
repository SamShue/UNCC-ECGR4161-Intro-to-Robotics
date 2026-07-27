#!/usr/bin/env python3
"""Generate a GIF showing Model Predictive Control (MPC) in action.

The script demonstrates receding-horizon control on a point robot with
unicycle dynamics navigating around circular obstacles.

At each control step:
1) sample many candidate control sequences over a short horizon,
2) score each sequence with a cost function,
3) apply only the first control from the best sequence,
4) repeat from the updated state.

Output:
    model_predictive_control_outputs/mpc_navigation.gif
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Polygon, Rectangle


# ---------------------------------------------------------------------------
# World and robot settings
# ---------------------------------------------------------------------------
WORLD_X_MIN = 0.0
WORLD_X_MAX = 10.0
WORLD_Y_MIN = 0.0
WORLD_Y_MAX = 10.0

START_STATE = np.array([1.2, 1.2, 0.0], dtype=float)  # x, y, theta(rad)
GOAL_POSITION = np.array([8.9, 8.7], dtype=float)

ROBOT_RADIUS = 0.22
GOAL_TOLERANCE = 0.35

# Interior wall rectangles (xmin, xmax, ymin, ymax), arranged to form rooms
# with doorway gaps so the robot must bend through openings.
ROOM_WALLS = [
    # Left/middle partition with two doorway openings.
    (3.2, 3.5, 0.0, 1.8),
    (3.2, 3.5, 3.0, 6.2),
    (3.2, 3.5, 7.4, 10.0),
    # Right-side partition that creates an upper room with one doorway.
    (3.5, 6.2, 5.0, 5.3),
    (7.0, 10.0, 5.0, 5.3),
    # Small divider in the lower-right area to add another turn.
    (6.1, 6.4, 0.0, 3.4),
]

OBSTACLES = [
    (np.array([1.8, 4.4], dtype=float), 0.58),
    (np.array([5.4, 2.2], dtype=float), 0.48),
    (np.array([7.8, 7.2], dtype=float), 0.44),
    (np.array([6.0, 8.5], dtype=float), 0.36),
]

# Global waypoints represent a map-only corridor route through doorways.
# Circular obstacles are local objects and are intentionally excluded from
# this global guidance sequence.
GLOBAL_WAYPOINTS = [
    np.array([3.35, 2.35], dtype=float),
    np.array([5.35, 3.95], dtype=float),
    np.array([6.60, 5.10], dtype=float),
    np.array([7.45, 6.45], dtype=float),
]


# ---------------------------------------------------------------------------
# MPC settings
# ---------------------------------------------------------------------------
DT = 0.18
HORIZON_STEPS = 12
MAX_MPC_STEPS = 140

LINEAR_OPTIONS = np.array([-0.25, 0.00, 0.30, 0.65, 0.95, 1.25], dtype=float)
ANGULAR_OPTIONS = np.array([-1.4, -1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0, 1.4], dtype=float)
ROLLOUT_SAMPLES = 320

W_GOAL = 1.00
W_CLEARANCE = 1.40
W_CONTROL_SMOOTH = 0.08
W_HEADING = 0.18
W_PROGRESS = 0.85
WAYPOINT_TOLERANCE = 0.50
PREFERRED_CLEARANCE = 0.14
VISUAL_CANDIDATE_COUNT = 90


# ---------------------------------------------------------------------------
# Output and style
# ---------------------------------------------------------------------------
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "model_predictive_control_outputs"
GIF_NAME = "mpc_navigation.gif"
FPS = 6

BG_COLOR = "#F8FAFC"
BORDER_COLOR = "#334155"
GRID_COLOR = "#CBD5E1"
TEXT_COLOR = "#0F172A"
OBSTACLE_FACE = "#FCA5A5"
OBSTACLE_EDGE = "#B91C1C"
WALL_FACE = "#94A3B8"
WALL_EDGE = "#475569"
ROBOT_COLOR = "#2563EB"
ROBOT_EDGE = "#1E293B"
GOAL_COLOR = "#16A34A"
TRAIL_COLOR = "#0F766E"
CANDIDATE_COLOR = "#0EA5E9"
BEST_COLOR = "#F59E0B"


@dataclass
class Rollout:
    """One sampled control sequence and its predicted path/cost."""

    controls: list[tuple[float, float]]
    path: np.ndarray
    cost: float
    valid: bool
    min_clearance: float


@dataclass
class FrameState:
    """Renderable state for one animation frame."""

    state: np.ndarray
    trail: np.ndarray
    candidates: list[Rollout]
    best: Rollout | None
    step_index: int
    target_index: int
    target_position: np.ndarray


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def robot_triangle(state: np.ndarray, size: float = 0.38) -> np.ndarray:
    """Return triangle vertices for a robot glyph at (x, y, theta)."""
    x, y, theta = state
    tip = np.array([x, y]) + size * np.array([np.cos(theta), np.sin(theta)])
    left = np.array([x, y]) + size * np.array([np.cos(theta + 2.45), np.sin(theta + 2.45)])
    right = np.array([x, y]) + size * np.array([np.cos(theta - 2.45), np.sin(theta - 2.45)])
    return np.vstack([tip, left, right])


def step_dynamics(state: np.ndarray, linear: float, angular: float) -> np.ndarray:
    """Advance unicycle dynamics one time step."""
    x, y, theta = state
    theta_next = wrap_angle(theta + angular * DT)
    x_next = x + linear * np.cos(theta_next) * DT
    y_next = y + linear * np.sin(theta_next) * DT
    return np.array([x_next, y_next, theta_next], dtype=float)


def clearance_to_obstacles(point_xy: np.ndarray) -> float:
    """Signed distance from robot boundary to nearest obstacle boundary."""
    values = []
    for center, radius in OBSTACLES:
        d = float(np.linalg.norm(point_xy - center)) - (radius + ROBOT_RADIUS)
        values.append(d)

    for xmin, xmax, ymin, ymax in ROOM_WALLS:
        d = signed_distance_to_rect(point_xy, xmin, xmax, ymin, ymax) - ROBOT_RADIUS
        values.append(d)

    return min(values) if values else np.inf


def signed_distance_to_rect(
    point_xy: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> float:
    """Return signed distance from a point to an axis-aligned rectangle.

    Positive outside, negative inside.
    """
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    hx = 0.5 * (xmax - xmin)
    hy = 0.5 * (ymax - ymin)

    qx = abs(point_xy[0] - cx) - hx
    qy = abs(point_xy[1] - cy) - hy
    outside = np.hypot(max(qx, 0.0), max(qy, 0.0))
    inside = min(max(qx, qy), 0.0)
    return float(outside + inside)


def out_of_bounds(point_xy: np.ndarray) -> bool:
    """Check if robot center would leave world bounds (with radius margin)."""
    x, y = point_xy
    return (
        x < WORLD_X_MIN + ROBOT_RADIUS
        or x > WORLD_X_MAX - ROBOT_RADIUS
        or y < WORLD_Y_MIN + ROBOT_RADIUS
        or y > WORLD_Y_MAX - ROBOT_RADIUS
    )


def sample_control_sequence(
    rng: np.random.Generator,
    state: np.ndarray,
    target_position: np.ndarray,
    mode: str,
) -> list[tuple[float, float]]:
    """Draw one control sequence for the MPC horizon.

    Different sampling modes intentionally produce diverse candidate families:
    target-seeking, random exploration, zig-zag exploration, and recovery arcs.
    """
    sequence: list[tuple[float, float]] = []
    simulated = state.copy()
    for _ in range(HORIZON_STEPS):
        if mode == "guided":
            to_target = target_position - simulated[:2]
            desired_heading = float(np.arctan2(to_target[1], to_target[0]))
            heading_error = wrap_angle(desired_heading - float(simulated[2]))
            turn_command = np.clip(1.25 * heading_error, ANGULAR_OPTIONS.min(), ANGULAR_OPTIONS.max())
            turn_command += float(rng.normal(0.0, 0.16))
            angular = float(ANGULAR_OPTIONS[np.argmin(np.abs(ANGULAR_OPTIONS - turn_command))])

            if abs(heading_error) < 0.35:
                linear = float(rng.choice([0.95, 1.25]))
            elif abs(heading_error) < 0.80:
                linear = float(rng.choice([0.65, 0.95]))
            else:
                linear = float(rng.choice([-0.25, 0.00, 0.30, 0.65]))
        elif mode == "zigzag":
            sign = -1.0 if rng.random() < 0.5 else 1.0
            angular = float(rng.choice([0.6, 1.0, 1.4])) * sign
            linear = float(rng.choice([0.30, 0.65, 0.95]))
        elif mode == "pivot":
            angular = float(rng.choice([-1.4, -1.0, 1.0, 1.4]))
            linear = float(rng.choice([-0.25, 0.00, 0.30]))
        else:
            linear = float(rng.choice(LINEAR_OPTIONS))
            angular = float(rng.choice(ANGULAR_OPTIONS))

        sequence.append((linear, angular))
        simulated = step_dynamics(simulated, linear, angular)
    return sequence


def rollout_signature(rollout: Rollout) -> np.ndarray:
    """Return a compact feature vector used for visual diversity selection."""
    end_x, end_y, end_theta = rollout.path[-1]
    total_turn = 0.0
    for (_, w0), (_, w1) in zip(rollout.controls, rollout.controls[1:]):
        total_turn += abs(w0) + abs(w1)
    return np.array([end_x, end_y, end_theta, total_turn], dtype=float)


def pick_diverse_rollouts(candidates: list[Rollout], count: int) -> list[Rollout]:
    """Pick a visually diverse subset of rollouts for plotting.

    Uses a greedy farthest-point strategy in rollout-signature space so plotted
    trajectories spread across distinct shapes and end points.
    """
    if not candidates:
        return []

    if len(candidates) <= count:
        return candidates

    signatures = np.array([rollout_signature(rollout) for rollout in candidates])
    scales = np.std(signatures, axis=0)
    scales[scales < 1e-6] = 1.0
    normalized = signatures / scales

    selected_indices: list[int] = []
    seed_index = int(np.argmin([rollout.cost for rollout in candidates]))
    selected_indices.append(seed_index)

    min_dist = np.linalg.norm(normalized - normalized[seed_index], axis=1)
    while len(selected_indices) < count:
        next_index = int(np.argmax(min_dist))
        if next_index in selected_indices:
            break
        selected_indices.append(next_index)
        dist_to_new = np.linalg.norm(normalized - normalized[next_index], axis=1)
        min_dist = np.minimum(min_dist, dist_to_new)

    return [candidates[index] for index in selected_indices]


def rollout_cost(
    start_state: np.ndarray,
    path: np.ndarray,
    controls: list[tuple[float, float]],
    min_clearance: float,
    target_position: np.ndarray,
) -> float:
    """Compute weighted MPC objective for one predicted trajectory."""
    final_xy = path[-1, :2]
    final_theta = float(path[-1, 2])

    start_distance = float(np.linalg.norm(start_state[:2] - target_position))
    goal_distance = float(np.linalg.norm(final_xy - target_position))
    progress = start_distance - goal_distance
    heading_to_goal = np.arctan2(
        target_position[1] - final_xy[1],
        target_position[0] - final_xy[0],
    )
    heading_error = abs(wrap_angle(heading_to_goal - final_theta))

    smoothness = 0.0
    for (v0, w0), (v1, w1) in zip(controls, controls[1:]):
        smoothness += abs(v1 - v0) + 0.6 * abs(w1 - w0)

    # Penalize only when running tighter than the preferred safety margin.
    clearance_slack = max(PREFERRED_CLEARANCE - min_clearance, 0.0)
    clearance_term = clearance_slack * clearance_slack

    return (
        W_GOAL * goal_distance
        + W_CLEARANCE * clearance_term
        + W_CONTROL_SMOOTH * smoothness
        + W_HEADING * heading_error
        - W_PROGRESS * progress
    )


def evaluate_sequence(
    state: np.ndarray,
    controls: list[tuple[float, float]],
    target_position: np.ndarray,
) -> Rollout:
    """Simulate and score one candidate sequence."""
    simulated = state.copy()
    path = [simulated.copy()]
    min_clearance = np.inf
    valid = True

    for linear, angular in controls:
        simulated = step_dynamics(simulated, linear, angular)
        path.append(simulated.copy())

        point_xy = simulated[:2]
        if out_of_bounds(point_xy):
            valid = False
        clearance = clearance_to_obstacles(point_xy)
        min_clearance = min(min_clearance, clearance)
        if clearance < 0.0:
            valid = False

    path_array = np.array(path)
    cost = rollout_cost(state, path_array, controls, min_clearance, target_position)
    if not valid:
        cost += 150.0

    return Rollout(
        controls=controls,
        path=path_array,
        cost=cost,
        valid=valid,
        min_clearance=float(min_clearance),
    )


def run_mpc_navigation(seed: int = 7) -> list[FrameState]:
    """Run receding-horizon MPC and return animation states."""
    rng = np.random.default_rng(seed)
    navigation_targets = [*GLOBAL_WAYPOINTS, GOAL_POSITION]

    state = START_STATE.copy()
    trail = [state[:2].copy()]
    frames: list[FrameState] = []
    target_index = 0

    for step in range(MAX_MPC_STEPS):
        while (
            target_index < len(navigation_targets) - 1
            and float(np.linalg.norm(state[:2] - navigation_targets[target_index]))
            <= WAYPOINT_TOLERANCE
        ):
            target_index += 1

        target_position = navigation_targets[target_index]
        candidates: list[Rollout] = []
        guided_samples = int(0.48 * ROLLOUT_SAMPLES)
        random_samples = int(0.26 * ROLLOUT_SAMPLES)
        zigzag_samples = int(0.16 * ROLLOUT_SAMPLES)
        pivot_samples = ROLLOUT_SAMPLES - guided_samples - random_samples - zigzag_samples

        for _ in range(guided_samples):
            controls = sample_control_sequence(rng, state, target_position, mode="guided")
            candidates.append(evaluate_sequence(state, controls, target_position))

        for _ in range(random_samples):
            controls = sample_control_sequence(rng, state, target_position, mode="random")
            candidates.append(evaluate_sequence(state, controls, target_position))

        for _ in range(zigzag_samples):
            controls = sample_control_sequence(rng, state, target_position, mode="zigzag")
            candidates.append(evaluate_sequence(state, controls, target_position))

        for _ in range(pivot_samples):
            controls = sample_control_sequence(rng, state, target_position, mode="pivot")
            candidates.append(evaluate_sequence(state, controls, target_position))

        valid = [candidate for candidate in candidates if candidate.valid]
        best = min(valid, key=lambda r: r.cost) if valid else None

        frames.append(
            FrameState(
                state=state.copy(),
                trail=np.array(trail, dtype=float),
                candidates=candidates,
                best=best,
                step_index=step,
                target_index=target_index,
                target_position=target_position.copy(),
            )
        )

        if best is None:
            break

        # Receding-horizon action: apply only first control, then re-optimize.
        linear_0, angular_0 = best.controls[0]
        state = step_dynamics(state, linear_0, angular_0)
        trail.append(state[:2].copy())

        if float(np.linalg.norm(state[:2] - GOAL_POSITION)) <= GOAL_TOLERANCE:
            frames.append(
                FrameState(
                    state=state.copy(),
                    trail=np.array(trail, dtype=float),
                    candidates=candidates,
                    best=best,
                    step_index=step + 1,
                    target_index=target_index,
                    target_position=target_position.copy(),
                )
            )
            break

    return frames


def draw_world(axis: plt.Axes) -> None:
    """Draw world bounds, room walls, circular obstacles, and goal."""
    axis.clear()
    axis.set_xlim(WORLD_X_MIN, WORLD_X_MAX)
    axis.set_ylim(WORLD_Y_MIN, WORLD_Y_MAX)
    axis.set_aspect("equal")
    axis.set_facecolor(BG_COLOR)

    for spine in axis.spines.values():
        spine.set_linewidth(1.4)
        spine.set_color(BORDER_COLOR)

    axis.grid(True, color=GRID_COLOR, linewidth=0.7, alpha=0.8)
    axis.set_xticks(np.arange(WORLD_X_MIN, WORLD_X_MAX + 0.1, 1.0))
    axis.set_yticks(np.arange(WORLD_Y_MIN, WORLD_Y_MAX + 0.1, 1.0))
    axis.tick_params(labelsize=8, colors=BORDER_COLOR)

    for xmin, xmax, ymin, ymax in ROOM_WALLS:
        axis.add_patch(
            Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                facecolor=WALL_FACE,
                edgecolor=WALL_EDGE,
                linewidth=1.2,
                alpha=0.95,
                zorder=2,
            )
        )

    for center, radius in OBSTACLES:
        axis.add_patch(
            Circle(
                center,
                radius,
                facecolor=OBSTACLE_FACE,
                edgecolor=OBSTACLE_EDGE,
                linewidth=1.5,
                alpha=0.9,
                zorder=3,
            )
        )

    axis.scatter(
        GOAL_POSITION[0],
        GOAL_POSITION[1],
        s=180,
        marker="*",
        color=GOAL_COLOR,
        edgecolor="#14532D",
        linewidth=1.0,
        zorder=8,
    )

    waypoint_x = [point[0] for point in GLOBAL_WAYPOINTS]
    waypoint_y = [point[1] for point in GLOBAL_WAYPOINTS]
    if waypoint_x:
        axis.scatter(
            waypoint_x,
            waypoint_y,
            s=34,
            marker="o",
            color="#7C3AED",
            edgecolor="white",
            linewidth=0.8,
            alpha=0.9,
            zorder=8,
        )


def draw_frame(axis: plt.Axes, frame: FrameState, total_frames: int) -> None:
    """Render one MPC state into the axis."""
    draw_world(axis)

    if len(frame.trail) > 1:
        axis.plot(
            frame.trail[:, 0],
            frame.trail[:, 1],
            color=TRAIL_COLOR,
            linewidth=2.4,
            alpha=0.95,
            zorder=6,
        )

    plotted = pick_diverse_rollouts(frame.candidates, VISUAL_CANDIDATE_COUNT)
    for rollout in plotted:
        if rollout.valid:
            axis.plot(
                rollout.path[:, 0],
                rollout.path[:, 1],
                color=CANDIDATE_COLOR,
                linewidth=0.9,
                linestyle=":",
                alpha=0.26,
                zorder=4,
            )
        else:
            axis.plot(
                rollout.path[:, 0],
                rollout.path[:, 1],
                color="#DC2626",
                linewidth=0.8,
                linestyle="--",
                alpha=0.12,
                zorder=3,
            )

    if frame.best is not None:
        axis.plot(
            frame.best.path[:, 0],
            frame.best.path[:, 1],
            color=BEST_COLOR,
            linewidth=2.6,
            alpha=0.95,
            zorder=7,
        )

    robot_poly = Polygon(
        robot_triangle(frame.state),
        closed=True,
        facecolor=ROBOT_COLOR,
        edgecolor=ROBOT_EDGE,
        linewidth=1.2,
        zorder=9,
    )
    axis.add_patch(robot_poly)
    axis.add_patch(
        Circle(
            frame.state[:2],
            ROBOT_RADIUS,
            facecolor="none",
            edgecolor=ROBOT_EDGE,
            linewidth=1.0,
            alpha=0.8,
            zorder=9,
        )
    )

    axis.scatter(
        frame.target_position[0],
        frame.target_position[1],
        s=68,
        marker="D",
        color="#7C3AED",
        edgecolor="white",
        linewidth=0.9,
        alpha=0.95,
        zorder=9,
    )

    goal_distance = float(np.linalg.norm(frame.state[:2] - GOAL_POSITION))
    target_distance = float(np.linalg.norm(frame.state[:2] - frame.target_position))
    best_cost = frame.best.cost if frame.best is not None else float("nan")
    valid_count = sum(1 for rollout in frame.candidates if rollout.valid)

    axis.set_title("Model Predictive Control in Room-Like Map", fontsize=13, weight="bold", pad=9)
    axis.text(
        0.02,
        0.02,
        f"Step: {frame.step_index + 1}/{max(total_frames, 1)}\n"
        f"Active global target: {frame.target_index + 1}/{len(GLOBAL_WAYPOINTS) + 1}\n"
        f"Distance to active target: {target_distance:.2f}\n"
        f"Valid candidates: {valid_count}/{len(frame.candidates)}\n"
        f"Best cost: {best_cost:.2f}\n"
        f"Distance to goal: {goal_distance:.2f}",
        transform=axis.transAxes,
        fontsize=8.5,
        color=TEXT_COLOR,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.92},
    )


def build_mpc_gif(output_path: Path) -> None:
    """Create and save the MPC navigation GIF."""
    frames = run_mpc_navigation()

    figure, axis = plt.subplots(figsize=(7.2, 7.2))

    def _update(frame_index: int) -> list:
        draw_frame(axis, frames[frame_index], len(frames))
        return []

    animation = FuncAnimation(
        figure,
        _update,
        frames=len(frames),
        interval=1000.0 / FPS,
        blit=False,
        repeat=True,
    )

    animation.save(output_path, writer=PillowWriter(fps=FPS), dpi=100)
    plt.close(figure)


def main() -> None:
    """Generate the MPC animation GIF."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    gif_path = OUTPUT_DIRECTORY / GIF_NAME
    build_mpc_gif(gif_path)
    print(f"Saved {gif_path}")


if __name__ == "__main__":
    main()