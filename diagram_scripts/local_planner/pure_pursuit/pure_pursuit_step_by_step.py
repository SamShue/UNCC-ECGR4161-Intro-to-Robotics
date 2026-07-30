#!/usr/bin/env python3
"""Generate step-by-step pure pursuit diagrams.

This script exports six PNGs showing one pure pursuit update cycle:
1) Starting state.
2) Lookahead circle and target point on the active segment.
3) Heading line/angle to target.
4) Curvature circle implied by pure pursuit.
5) Robot moved forward along the commanded curvature.
6) Start over from the new pose with a new lookahead target.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, Polygon


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WORLD_MIN = 0.0
WORLD_MAX = 10.0

PATH_POINTS = np.array(
    [
        [1.2, 1.4],   # P0
        [6.1, 2.5],   # P1
        [8.4, 8.2],   # P2
    ],
    dtype=float,
)

ROBOT_START = np.array([3.1, 3.0], dtype=float)
ROBOT_HEADING_DEG = 10.0

LOOKAHEAD_DISTANCE = 1.55
FORWARD_STEP = 1.0

OUTPUT_DIR = Path(__file__).resolve().parent / "pure_pursuit_outputs"

# Visual style
BG_COLOR = "#F8FAFC"
GRID_COLOR = "#CBD5E1"
BORDER_COLOR = "#334155"
TEXT_COLOR = "#0F172A"

PATH_COLOR = "#0EA5E9"
PATH_FUTURE_COLOR = "#9CA3AF"
ROBOT_COLOR = "#2563EB"
ROBOT_EDGE = "#1E293B"
LOOKAHEAD_COLOR = "#F59E0B"
TARGET_COLOR = "#EF4444"
CURVATURE_COLOR = "#8B5CF6"
HEADING_RAY_COLOR = "#EF4444"


@dataclass
class PursuitState:
    """State used by one pure pursuit update step."""

    robot_xy: np.ndarray
    robot_heading_rad: float
    segment_index: int
    target_xy: np.ndarray
    alpha_rad: float
    curvature: float


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def wrap_angle_rad(angle: float) -> float:
    """Wrap to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def robot_triangle(position: np.ndarray, heading_rad: float, size: float = 0.38) -> np.ndarray:
    """Build a triangular robot glyph aligned with heading."""
    tip = position + size * np.array([np.cos(heading_rad), np.sin(heading_rad)])
    left = position + size * np.array([np.cos(heading_rad + 2.45), np.sin(heading_rad + 2.45)])
    right = position + size * np.array([np.cos(heading_rad - 2.45), np.sin(heading_rad - 2.45)])
    return np.array([tip, left, right])


def project_point_to_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """Project a point to a segment and return projection point and segment t in [0, 1]."""
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return a.copy(), 0.0

    t = float(np.dot(point - a, ab) / denom)
    t_clamped = min(1.0, max(0.0, t))
    projection = a + t_clamped * ab
    return projection, t_clamped


def nearest_segment_index(point: np.ndarray, polyline: np.ndarray) -> int:
    """Return index i for segment [i, i+1] nearest to point."""
    best_index = 0
    best_dist = float("inf")

    for i in range(len(polyline) - 1):
        projection, _ = project_point_to_segment(point, polyline[i], polyline[i + 1])
        dist = float(np.linalg.norm(point - projection))
        if dist < best_dist:
            best_dist = dist
            best_index = i

    return best_index


def circle_segment_intersections(
    center: np.ndarray,
    radius: float,
    a: np.ndarray,
    b: np.ndarray,
    t_min: float = 0.0,
) -> list[tuple[float, np.ndarray]]:
    """Intersect a circle with one line segment.

    Returns list of (t, point) with t in [0, 1] and t >= t_min.
    """
    d = b - a
    f = a - center

    aa = float(np.dot(d, d))
    bb = 2.0 * float(np.dot(f, d))
    cc = float(np.dot(f, f) - radius * radius)

    discriminant = bb * bb - 4.0 * aa * cc
    if aa < 1e-12 or discriminant < 0.0:
        return []

    root = np.sqrt(discriminant)
    t1 = (-bb - root) / (2.0 * aa)
    t2 = (-bb + root) / (2.0 * aa)

    hits: list[tuple[float, np.ndarray]] = []
    for t in (t1, t2):
        if t_min - 1e-9 <= t <= 1.0 + 1e-9:
            t_clip = min(1.0, max(0.0, t))
            hits.append((t_clip, a + t_clip * d))

    hits.sort(key=lambda item: item[0])
    return hits


def find_lookahead_target(
    robot_xy: np.ndarray,
    polyline: np.ndarray,
    lookahead_distance: float,
) -> tuple[int, np.ndarray]:
    """Find lookahead target point on current-or-later polyline segments."""
    start_seg = nearest_segment_index(robot_xy, polyline)

    for i in range(start_seg, len(polyline) - 1):
        a = polyline[i]
        b = polyline[i + 1]

        t_min = 0.0
        if i == start_seg:
            _, t_near = project_point_to_segment(robot_xy, a, b)
            t_min = t_near

        hits = circle_segment_intersections(robot_xy, lookahead_distance, a, b, t_min=t_min)
        if hits:
            return i, hits[-1][1]

    return len(polyline) - 2, polyline[-1].copy()


def compute_pursuit_state(
    robot_xy: np.ndarray,
    robot_heading_rad: float,
    polyline: np.ndarray,
    lookahead_distance: float,
) -> PursuitState:
    """Compute target point, heading error, and curvature command."""
    seg_index, target = find_lookahead_target(robot_xy, polyline, lookahead_distance)

    target_heading = float(np.arctan2(target[1] - robot_xy[1], target[0] - robot_xy[0]))
    alpha = wrap_angle_rad(target_heading - robot_heading_rad)
    curvature = 2.0 * np.sin(alpha) / lookahead_distance

    return PursuitState(
        robot_xy=robot_xy.copy(),
        robot_heading_rad=robot_heading_rad,
        segment_index=seg_index,
        target_xy=target,
        alpha_rad=alpha,
        curvature=curvature,
    )


def advance_unicycle(robot_xy: np.ndarray, heading_rad: float, curvature: float, ds: float) -> tuple[np.ndarray, float]:
    """Move robot forward by ds using constant curvature."""
    if abs(curvature) < 1e-8:
        new_xy = robot_xy + ds * np.array([np.cos(heading_rad), np.sin(heading_rad)])
        return new_xy, heading_rad

    radius = 1.0 / curvature
    cx = robot_xy[0] - radius * np.sin(heading_rad)
    cy = robot_xy[1] + radius * np.cos(heading_rad)

    dtheta = curvature * ds
    new_heading = heading_rad + dtheta

    new_x = cx + radius * np.sin(new_heading)
    new_y = cy - radius * np.cos(new_heading)

    return np.array([new_x, new_y], dtype=float), new_heading


def curvature_circle_center(robot_xy: np.ndarray, heading_rad: float, curvature: float) -> tuple[np.ndarray, float] | None:
    """Return center and radius of turning circle if curvature is nonzero."""
    if abs(curvature) < 1e-8:
        return None

    radius = abs(1.0 / curvature)
    left_normal = np.array([-np.sin(heading_rad), np.cos(heading_rad)])
    direction = np.sign(curvature)
    center = robot_xy + direction * radius * left_normal
    return center, radius


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def setup_axes(axis: plt.Axes, title: str, subtitle: str) -> None:
    """Apply consistent style to a frame."""
    axis.set_xlim(WORLD_MIN, WORLD_MAX)
    axis.set_ylim(WORLD_MIN, WORLD_MAX)
    axis.set_aspect("equal")
    axis.set_facecolor(BG_COLOR)

    for spine in axis.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color(BORDER_COLOR)

    axis.grid(True, color=GRID_COLOR, linewidth=0.75, alpha=0.8)
    axis.set_xticks(np.arange(WORLD_MIN, WORLD_MAX + 0.1, 1.0))
    axis.set_yticks(np.arange(WORLD_MIN, WORLD_MAX + 0.1, 1.0))
    axis.tick_params(colors=BORDER_COLOR, labelsize=8)

    axis.set_title(title, fontsize=14, weight="bold", color=TEXT_COLOR, pad=6)
    axis.text(
        0.02,
        0.02,
        subtitle,
        transform=axis.transAxes,
        fontsize=9,
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.25", fc="#F1F5F9", ec="#CBD5E1", alpha=0.95),
    )


def draw_path(axis: plt.Axes, active_segment: int | None = None) -> None:
    """Draw polyline path and waypoint labels."""
    axis.plot(PATH_POINTS[:, 0], PATH_POINTS[:, 1], color=PATH_COLOR, linewidth=2.4, zorder=2)

    if active_segment is not None and active_segment + 1 < len(PATH_POINTS):
        future = PATH_POINTS[active_segment + 1 :]
        if len(future) > 1:
            axis.plot(future[:, 0], future[:, 1], color=PATH_FUTURE_COLOR, linewidth=2.4, alpha=0.8, zorder=1)

    axis.scatter(PATH_POINTS[:, 0], PATH_POINTS[:, 1], s=28, color="#64748B", zorder=3)
    for idx, point in enumerate(PATH_POINTS):
        axis.text(point[0] + 0.12, point[1] + 0.12, f"P{idx}", fontsize=10, weight="bold", color=TEXT_COLOR)


def draw_robot(axis: plt.Axes, robot_xy: np.ndarray, heading_rad: float) -> None:
    """Draw robot triangle and center marker."""
    triangle = Polygon(
        robot_triangle(robot_xy, heading_rad),
        closed=True,
        facecolor=ROBOT_COLOR,
        edgecolor=ROBOT_EDGE,
        linewidth=1.1,
        zorder=8,
    )
    axis.add_patch(triangle)
    axis.scatter(robot_xy[0], robot_xy[1], s=22, color=ROBOT_EDGE, zorder=9)


def draw_heading_arc(axis: plt.Axes, state: PursuitState, radius: float = 0.85) -> None:
    """Draw heading error arc between current heading and target bearing."""
    heading_deg = np.degrees(state.robot_heading_rad)
    target_deg = np.degrees(state.robot_heading_rad + state.alpha_rad)

    arc = Arc(
        xy=state.robot_xy,
        width=2.0 * radius,
        height=2.0 * radius,
        theta1=heading_deg,
        theta2=target_deg,
        color="#DC2626",
        linewidth=1.6,
        zorder=6,
    )
    axis.add_patch(arc)

    mid = state.robot_heading_rad + 0.5 * state.alpha_rad
    label_xy = state.robot_xy + (radius + 0.2) * np.array([np.cos(mid), np.sin(mid)])
    axis.text(
        label_xy[0],
        label_xy[1],
        r"$\alpha$",
        color="#B91C1C",
        fontsize=13,
        weight="bold",
        ha="center",
        va="center",
    )


def draw_curvature_circle(axis: plt.Axes, state: PursuitState) -> None:
    """Draw pure-pursuit turning circle and center point."""
    circle_info = curvature_circle_center(state.robot_xy, state.robot_heading_rad, state.curvature)
    if circle_info is None:
        return

    center, radius = circle_info
    turn_circle = Circle(
        center,
        radius,
        fill=False,
        linestyle="--",
        linewidth=1.9,
        edgecolor=CURVATURE_COLOR,
        alpha=0.95,
        zorder=1,
    )
    axis.add_patch(turn_circle)

    axis.scatter(center[0], center[1], s=18, color=CURVATURE_COLOR, zorder=7)
    axis.text(
        center[0] + 0.12,
        center[1] - 0.2,
        "C",
        fontsize=10,
        color=CURVATURE_COLOR,
        weight="bold",
    )

    radius_text = f"R={radius:.2f}"
    axis.text(
        0.98,
        0.08,
        radius_text,
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=CURVATURE_COLOR,
        bbox=dict(boxstyle="round,pad=0.2", fc="#F5F3FF", ec="#DDD6FE", alpha=0.95),
    )


def draw_common(axis: plt.Axes, state: PursuitState, show_lookahead: bool = False) -> None:
    """Draw scene elements shared by multiple steps."""
    draw_path(axis, active_segment=state.segment_index)
    draw_robot(axis, state.robot_xy, state.robot_heading_rad)

    if show_lookahead:
        lookahead = Circle(
            state.robot_xy,
            LOOKAHEAD_DISTANCE,
            fill=False,
            linestyle="--",
            linewidth=1.7,
            edgecolor=LOOKAHEAD_COLOR,
            zorder=4,
        )
        axis.add_patch(lookahead)

    axis.scatter(
        state.target_xy[0],
        state.target_xy[1],
        s=65,
        color=TARGET_COLOR,
        edgecolor="#7F1D1D",
        linewidth=0.9,
        zorder=9,
    )


def make_step_figure(step_id: int, state: PursuitState, moved_state: PursuitState | None = None) -> None:
    """Render and save one requested step diagram."""
    fig, axis = plt.subplots(figsize=(8.6, 8.1), dpi=120)

    if step_id == 1:
        setup_axes(
            axis,
            "Step 1: Starting State",
            "Robot has a pose and active path segment; next step is to form the lookahead circle.",
        )
        draw_path(axis, active_segment=state.segment_index)
        draw_robot(axis, state.robot_xy, state.robot_heading_rad)

    elif step_id == 2:
        setup_axes(
            axis,
            "Step 2: Lookahead Circle and Target",
            "Circle radius is lookahead distance Ld. Intersection with path gives the target point.",
        )
        draw_common(axis, state, show_lookahead=True)

    elif step_id == 3:
        setup_axes(
            axis,
            "Step 3: Heading to Lookahead Point",
            "Compute heading error alpha between robot heading and the lookahead bearing.",
        )
        draw_common(axis, state, show_lookahead=True)
        ray_end = state.robot_xy + 2.2 * np.array([
            np.cos(state.robot_heading_rad),
            np.sin(state.robot_heading_rad),
        ])
        axis.plot(
            [state.robot_xy[0], ray_end[0]],
            [state.robot_xy[1], ray_end[1]],
            linestyle="-",
            color="#1D4ED8",
            linewidth=1.6,
            alpha=0.85,
            zorder=5,
        )
        axis.plot(
            [state.robot_xy[0], state.target_xy[0]],
            [state.robot_xy[1], state.target_xy[1]],
            linestyle=":",
            color=HEADING_RAY_COLOR,
            linewidth=1.8,
            zorder=5,
        )
        draw_heading_arc(axis, state)

    elif step_id == 4:
        setup_axes(
            axis,
            "Step 4: Curvature Circle",
            "Pure pursuit curvature kappa = 2*sin(alpha)/Ld defines the instantaneous turning circle.",
        )
        draw_common(axis, state, show_lookahead=True)
        draw_curvature_circle(axis, state)

    elif step_id == 5 and moved_state is not None:
        setup_axes(
            axis,
            "Step 5: Robot Moves Forward",
            "Advance the robot by a short arc length using the computed curvature command.",
        )
        draw_path(axis, active_segment=state.segment_index)
        axis.scatter(state.robot_xy[0], state.robot_xy[1], s=28, color="#93C5FD", zorder=5)
        axis.plot(
            [state.robot_xy[0], moved_state.robot_xy[0]],
            [state.robot_xy[1], moved_state.robot_xy[1]],
            linestyle="--",
            color="#3B82F6",
            linewidth=1.4,
            zorder=4,
        )
        draw_robot(axis, moved_state.robot_xy, moved_state.robot_heading_rad)
        axis.text(
            moved_state.robot_xy[0] + 0.15,
            moved_state.robot_xy[1] - 0.35,
            "new pose",
            fontsize=9,
            color="#1E3A8A",
            weight="bold",
        )

    elif step_id == 6 and moved_state is not None:
        setup_axes(
            axis,
            "Step 6: Start Over at New Pose",
            "From the updated pose, build a new lookahead circle and pick the next target point.",
        )
        draw_common(axis, moved_state, show_lookahead=True)

    else:
        plt.close(fig)
        raise ValueError(f"Unsupported step id: {step_id}")

    out_path = OUTPUT_DIR / f"pure_pursuit_step_{step_id:02d}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> None:
    """Compute one pure pursuit cycle and export all step images."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    heading_start_rad = np.deg2rad(ROBOT_HEADING_DEG)
    first_state = compute_pursuit_state(
        robot_xy=ROBOT_START,
        robot_heading_rad=heading_start_rad,
        polyline=PATH_POINTS,
        lookahead_distance=LOOKAHEAD_DISTANCE,
    )

    new_xy, new_heading = advance_unicycle(
        robot_xy=first_state.robot_xy,
        heading_rad=first_state.robot_heading_rad,
        curvature=first_state.curvature,
        ds=FORWARD_STEP,
    )

    second_state = compute_pursuit_state(
        robot_xy=new_xy,
        robot_heading_rad=new_heading,
        polyline=PATH_POINTS,
        lookahead_distance=LOOKAHEAD_DISTANCE,
    )

    make_step_figure(1, first_state)
    make_step_figure(2, first_state)
    make_step_figure(3, first_state)
    make_step_figure(4, first_state)
    make_step_figure(5, first_state, moved_state=second_state)
    make_step_figure(6, first_state, moved_state=second_state)

    print(f"Saved 6 step images to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
