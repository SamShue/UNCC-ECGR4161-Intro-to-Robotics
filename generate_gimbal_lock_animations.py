#!/usr/bin/env python3
"""Generate GIF animations that explain Euler angles and gimbal lock.

Euler convention used in this lecture script:
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

- roll: rotation about robot/body x-axis
- pitch: rotation about intermediate/body y-axis
- yaw: rotation about world z-axis
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# Slide-friendly visual defaults.
FIGSIZE = (12.8, 7.2)  # 16:9
FPS = 24
FRAMES = 90
AXIS_LIMIT = 1.25


def Rx(angle: float) -> np.ndarray:
    """Rotation matrix about x-axis for angle in degrees."""
    a = np.deg2rad(angle)
    c, s = np.cos(a), np.sin(a)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ]
    )


def Ry(angle: float) -> np.ndarray:
    """Rotation matrix about y-axis for angle in degrees."""
    a = np.deg2rad(angle)
    c, s = np.cos(a), np.sin(a)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ]
    )


def Rz(angle: float) -> np.ndarray:
    """Rotation matrix about z-axis for angle in degrees."""
    a = np.deg2rad(angle)
    c, s = np.cos(a), np.sin(a)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def rotation_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Compose ZYX Euler rotations as Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    return Rz(yaw_deg) @ Ry(pitch_deg) @ Rx(roll_deg)


def draw_frame(ax, R: np.ndarray, prefix: str) -> None:
    """Draw a coordinate frame whose basis vectors are the columns of R."""
    colors = ["#d62728", "#2ca02c", "#1f77b4"]  # X, Y, Z
    labels = ["X", "Y", "Z"]

    for i in range(3):
        v = R[:, i]
        ax.quiver(
            0.0,
            0.0,
            0.0,
            0.95 * v[0],
            0.95 * v[1],
            0.95 * v[2],
            color=colors[i],
            linewidth=2.6,
            arrow_length_ratio=0.12,
        )
        tip = 1.08 * v
        ax.text(
            tip[0],
            tip[1],
            tip[2],
            f"{prefix} {labels[i]}",
            color=colors[i],
            fontsize=12,
            fontweight="bold",
        )


def draw_rotation_axis(ax, axis_vector: Sequence[float], label: str, color: str) -> None:
    """Draw a highlighted axis line through the origin and attach a label."""
    axis = np.asarray(axis_vector, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return
    axis = axis / norm

    p1 = -1.15 * axis
    p2 = 1.15 * axis
    ax.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        [p1[2], p2[2]],
        color=color,
        linewidth=3.0,
        alpha=0.95,
    )
    label_pos = 1.2 * axis
    ax.text(
        label_pos[0],
        label_pos[1],
        label_pos[2],
        label,
        color=color,
        fontsize=11,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
    )


def draw_simple_robot_model(ax, R: np.ndarray, center: Sequence[float] = (0.0, 0.0, 0.0)) -> None:
    """Draw a simple rigid-body robot model as a transformed cuboid.

    This keeps the visualization lightweight while giving students a clear
    body shape whose orientation changes with Euler angles.
    """
    c = np.asarray(center, dtype=float)

    # Body dimensions: length (x), width (y), height (z).
    lx, ly, lz = 0.70, 0.45, 0.20
    hx, hy, hz = lx / 2.0, ly / 2.0, lz / 2.0

    # Cuboid vertices in robot/body coordinates.
    local_vertices = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ]
    )

    # Rotate body into world coordinates and shift to center.
    world_vertices = (R @ local_vertices.T).T + c

    faces = [
        [world_vertices[i] for i in [0, 1, 2, 3]],  # bottom
        [world_vertices[i] for i in [4, 5, 6, 7]],  # top
        [world_vertices[i] for i in [0, 1, 5, 4]],
        [world_vertices[i] for i in [1, 2, 6, 5]],
        [world_vertices[i] for i in [2, 3, 7, 6]],
        [world_vertices[i] for i in [3, 0, 4, 7]],
    ]

    body = Poly3DCollection(
        faces,
        facecolors="#f2c14e",
        edgecolors="#333333",
        linewidths=1.0,
        alpha=0.65,
    )
    ax.add_collection3d(body)

    # Add a small heading marker on +X face to make front direction obvious.
    nose_local = np.array([[hx + 0.10, 0.0, 0.0]])
    nose_world = (R @ nose_local.T).T + c
    ax.scatter(
        nose_world[:, 0],
        nose_world[:, 1],
        nose_world[:, 2],
        color="#111111",
        s=35,
        depthshade=False,
    )


def setup_3d_axes(ax) -> None:
    """Apply consistent camera, limits, aspect ratio, and white background."""
    ax.set_xlim(-AXIS_LIMIT, AXIS_LIMIT)
    ax.set_ylim(-AXIS_LIMIT, AXIS_LIMIT)
    ax.set_zlim(-AXIS_LIMIT, AXIS_LIMIT)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=24, azim=-52)
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.25)

    ax.set_xlabel("X", fontsize=10, labelpad=6)
    ax.set_ylabel("Y", fontsize=10, labelpad=6)
    ax.set_zlabel("Z", fontsize=10, labelpad=6)
    ax.tick_params(labelsize=8)

    ax.xaxis.pane.set_facecolor((1, 1, 1, 1))
    ax.yaxis.pane.set_facecolor((1, 1, 1, 1))
    ax.zaxis.pane.set_facecolor((1, 1, 1, 1))


def interpolate_angles(
    start_angles: Sequence[float], end_angles: Sequence[float], t: float
) -> Tuple[float, float, float]:
    """Linear interpolation between two (roll, pitch, yaw) triplets."""
    a0 = np.asarray(start_angles, dtype=float)
    a1 = np.asarray(end_angles, dtype=float)
    a = (1.0 - t) * a0 + t * a1
    return float(a[0]), float(a[1]), float(a[2])


def create_animation(
    fig: plt.Figure,
    update_func: Callable[[int], list],
    frames: int,
    fps: int,
) -> FuncAnimation:
    """Create a looping FuncAnimation."""
    return FuncAnimation(
        fig,
        update_func,
        frames=frames,
        interval=1000.0 / fps,
        blit=False,
        repeat=True,
    )


def save_animation(anim: FuncAnimation, output_path: Path, fps: int) -> None:
    """Save a GIF using PillowWriter (no ffmpeg needed)."""
    writer = PillowWriter(fps=fps, metadata={"artist": "Robotics Lecture"})
    anim.save(str(output_path), writer=writer, dpi=100)


def _ease(frame_idx: int, n_frames: int) -> float:
    """Cosine easing for smooth start/stop interpolation."""
    if n_frames <= 1:
        return 1.0
    t = frame_idx / (n_frames - 1)
    return 0.5 - 0.5 * np.cos(np.pi * t)


def _rotation_distance_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Geodesic angle between two orientations in degrees."""
    Rrel = Ra.T @ Rb
    c = np.clip((np.trace(Rrel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(c)))


def _draw_world_and_robot(ax, R_robot: np.ndarray) -> None:
    draw_simple_robot_model(ax, R_robot)
    draw_frame(ax, np.eye(3), "World")
    draw_frame(ax, R_robot, "Robot")


def _draw_single_scene(
    ax,
    roll: float,
    pitch: float,
    yaw: float,
    title: str,
    message: str | None,
) -> None:
    ax.cla()
    setup_3d_axes(ax)

    R_robot = rotation_matrix(roll, pitch, yaw)
    _draw_world_and_robot(ax, R_robot)

    ax.set_title(title, fontsize=17, pad=14, fontweight="bold")
    ax.text2D(
        0.02,
        0.94,
        f"roll={roll:5.1f} deg   pitch={pitch:5.1f} deg   yaw={yaw:5.1f} deg",
        transform=ax.transAxes,
        fontsize=13,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 3},
    )
    if message:
        ax.text2D(
            0.02,
            0.04,
            message,
            transform=ax.transAxes,
            fontsize=14,
            bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "none", "pad": 3},
        )


def generate_single_transition(
    output_path: Path,
    title: str,
    start_angles: Tuple[float, float, float],
    end_angles: Tuple[float, float, float],
    frames: int,
    fps: int,
    message: str | None = None,
) -> None:
    """Generate one transition GIF for a single 3D plot."""
    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")

    def update(frame_idx: int) -> list:
        t = _ease(frame_idx, frames)
        roll, pitch, yaw = interpolate_angles(start_angles, end_angles, t)
        _draw_single_scene(ax, roll, pitch, yaw, title, message)
        return []

    anim = create_animation(fig, update, frames=frames, fps=fps)
    save_animation(anim, output_path, fps)
    plt.close(fig)


def generate_gimbal_lock(output_path: Path, frames: int, fps: int) -> None:
    """Generate side-by-side motion equivalence demo at pitch=90 deg.

    Why gimbal lock occurs here:
    when pitch reaches 90 deg in ZYX Euler angles, roll and yaw axes align,
    so roll and yaw can no longer provide independent rotations.
    """
    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    ax_left = fig.add_subplot(121, projection="3d", facecolor="white")
    ax_right = fig.add_subplot(122, projection="3d", facecolor="white")

    base_roll = 30.0
    base_pitch = 90.0
    base_yaw = 45.0

    def update(frame_idx: int) -> list:
        u = _ease(frame_idx, frames)
        delta = -30.0 + 60.0 * u

        # Left plot: vary roll, keep yaw fixed.
        r_l, p_l, y_l = base_roll + delta, base_pitch, base_yaw
        R_l = rotation_matrix(r_l, p_l, y_l)

        # Right plot: vary yaw oppositely, keep roll fixed.
        # At pitch=90 deg this produces the same orientation trajectory.
        r_r, p_r, y_r = base_roll, base_pitch, base_yaw - delta
        R_r = rotation_matrix(r_r, p_r, y_r)

        ax_left.cla()
        ax_right.cla()
        setup_3d_axes(ax_left)
        setup_3d_axes(ax_right)
        _draw_world_and_robot(ax_left, R_l)
        _draw_world_and_robot(ax_right, R_r)

        yaw_axis = np.array([0.0, 0.0, 1.0])
        draw_rotation_axis(ax_left, R_l[:, 0], "Roll axis (Robot X)", "#e377c2")
        draw_rotation_axis(ax_left, yaw_axis, "Yaw axis (World Z)", "#ff7f0e")
        draw_rotation_axis(ax_right, R_r[:, 0], "Roll axis (Robot X)", "#e377c2")
        draw_rotation_axis(ax_right, yaw_axis, "Yaw axis (World Z)", "#ff7f0e")

        ax_left.set_title("Left: Animate Roll, Fix Yaw", fontsize=13, pad=10, fontweight="bold")
        ax_right.set_title("Right: Animate Yaw, Fix Roll", fontsize=13, pad=10, fontweight="bold")

        alignment = abs(np.dot(R_l[:, 0] / np.linalg.norm(R_l[:, 0]), yaw_axis))
        diff = _rotation_distance_deg(R_l, R_r)

        fig.suptitle("Gimbal Lock at pitch = 90 deg", fontsize=19, y=0.98, fontweight="bold")
        fig.text(
            0.5,
            0.03,
            "Roll and yaw now describe the same rotational motion.",
            ha="center",
            fontsize=15,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none", "pad": 4},
        )
        fig.text(
            0.5,
            0.90,
            f"|dot(Robot X, World Z)|={alignment:.3f}   orientation difference={diff:.3f} deg",
            ha="center",
            fontsize=11,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 2},
        )
        return []

    anim = create_animation(fig, update, frames=frames, fps=fps)
    save_animation(anim, output_path, fps)
    plt.close(fig)


def _choose_ambiguous_pair() -> Tuple[Tuple[float, float, float], Tuple[float, float, float], float]:
    """Choose two near-lock Euler sets that represent the same orientation.

    Requested pair is tested first. If numerical mismatch is too large, fallback
    to a pair with equal (roll - yaw) at pitch=90, which preserves orientation.
    """
    pair_a = (20.0, 90.0, 10.0)
    pair_b = (30.0, 90.0, 20.0)

    diff = _rotation_distance_deg(rotation_matrix(*pair_a), rotation_matrix(*pair_b))
    if diff <= 1.0:
        return pair_a, pair_b, diff

    r_a, _, y_a = pair_a
    delta = r_a - y_a
    pair_b = (45.0, 90.0, 45.0 - delta)
    diff = _rotation_distance_deg(rotation_matrix(*pair_a), rotation_matrix(*pair_b))
    return pair_a, pair_b, diff


def generate_ambiguous_euler(output_path: Path, frames: int, fps: int) -> None:
    """Generate side-by-side static comparison of equivalent Euler sets."""
    pair_a, pair_b, diff = _choose_ambiguous_pair()

    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    ax_left = fig.add_subplot(121, projection="3d", facecolor="white")
    ax_right = fig.add_subplot(122, projection="3d", facecolor="white")

    def update(frame_idx: int) -> list:
        # Tiny pulse keeps GIF smooth while preserving a static orientation comparison.
        pulse = 0.985 + 0.015 * np.sin(2.0 * np.pi * frame_idx / max(frames - 1, 1))
        ring_t = np.linspace(0.0, 2.0 * np.pi, 80)

        ax_left.cla()
        ax_right.cla()
        setup_3d_axes(ax_left)
        setup_3d_axes(ax_right)

        R_left = rotation_matrix(*pair_a)
        R_right = rotation_matrix(*pair_b)
        _draw_world_and_robot(ax_left, R_left)
        _draw_world_and_robot(ax_right, R_right)

        ax_left.plot(pulse * 0.7 * np.cos(ring_t), pulse * 0.7 * np.sin(ring_t), 0.0 * ring_t, color="#999999", alpha=0.16)
        ax_right.plot(pulse * 0.7 * np.cos(ring_t), pulse * 0.7 * np.sin(ring_t), 0.0 * ring_t, color="#999999", alpha=0.16)

        ax_left.set_title(
            f"Left: roll={pair_a[0]:.0f}, pitch={pair_a[1]:.0f}, yaw={pair_a[2]:.0f}",
            fontsize=12,
            pad=10,
            fontweight="bold",
        )
        ax_right.set_title(
            f"Right: roll={pair_b[0]:.0f}, pitch={pair_b[1]:.0f}, yaw={pair_b[2]:.0f}",
            fontsize=12,
            pad=10,
            fontweight="bold",
        )

        fig.suptitle("Ambiguous Euler Angles Near Gimbal Lock", fontsize=18, y=0.98, fontweight="bold")
        fig.text(
            0.5,
            0.03,
            "Different Euler angle values can represent the same physical orientation.",
            ha="center",
            fontsize=15,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none", "pad": 4},
        )
        fig.text(
            0.5,
            0.90,
            f"Orientation difference between sets: {diff:.4f} deg",
            ha="center",
            fontsize=11,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 2},
        )
        return []

    anim = create_animation(fig, update, frames=frames, fps=fps)
    save_animation(anim, output_path, fps)
    plt.close(fig)


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "gimbal_lock_animations"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {out_dir}")
    print("Generating GIFs (this may take a minute)...")

    transitions = [
        (
            "initial_orientation.gif",
            "Initial Orientation (Robot Aligned with World)",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            None,
        ),
        (
            "yaw_rotation.gif",
            "Yaw Rotation About World Z",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 45.0),
            None,
        ),
        (
            "roll_rotation.gif",
            "Roll Rotation About Robot X",
            (0.0, 0.0, 45.0),
            (30.0, 0.0, 45.0),
            None,
        ),
        (
            "pitch_rotation.gif",
            "Pitch Rotation About Robot Y",
            (30.0, 0.0, 45.0),
            (30.0, 45.0, 45.0),
            None,
        ),
        (
            "approaching_gimbal_lock.gif",
            "Approaching Gimbal Lock",
            (30.0, 45.0, 45.0),
            (30.0, 80.0, 45.0),
            "Roll and yaw axes are becoming nearly aligned.",
        ),
        (
            "pitch_90_degrees.gif",
            "Pitch to 90 Degrees",
            (30.0, 80.0, 45.0),
            (30.0, 90.0, 45.0),
            "At pitch = 90 deg, roll and yaw align.",
        ),
    ]

    for idx, (name, title, start, end, msg) in enumerate(transitions, start=1):
        print(f"[{idx}/8] Generating {name} ...")
        generate_single_transition(
            output_path=out_dir / name,
            title=title,
            start_angles=start,
            end_angles=end,
            frames=FRAMES,
            fps=FPS,
            message=msg,
        )

    print("[7/8] Generating gimbal_lock.gif ...")
    generate_gimbal_lock(out_dir / "gimbal_lock.gif", frames=FRAMES, fps=FPS)

    print("[8/8] Generating ambiguous_euler_angles.gif ...")
    generate_ambiguous_euler(out_dir / "ambiguous_euler_angles.gif", frames=FRAMES, fps=FPS)

    print("Done. Generated files:")
    for p in sorted(out_dir.glob("*.gif")):
        print(f"  - {p.name} ({p.stat().st_size / (1024.0 * 1024.0):.2f} MB)")


if __name__ == "__main__":
    main()
