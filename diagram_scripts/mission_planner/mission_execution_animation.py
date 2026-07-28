#!/usr/bin/env python3
"""Animate a beer-fetch mission from a top-down 2D perspective.

The robot carries out the mission state machine (receive request, navigate to
the refrigerator, open it, find and grasp the beer, close the door, return to
the user, and deliver). Each frame highlights the state currently executing,
shows a cloud of localization particles around the robot, and reveals the
planned path toward the active goal as the planner "draws" it.

Output:
    mission_execution_outputs/beer_fetch_mission.gif
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle, Wedge


OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "mission_execution_outputs"
GIF_NAME = "beer_fetch_mission.gif"
FPS = 10

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
BG_COLOR = "#F8FAFC"
FLOOR_COLOR = "#FFFFFF"
BORDER_COLOR = "#334155"
TEXT_COLOR = "#0F172A"
MUTED_TEXT = "#475569"
ACCENT_BLUE = "#2563EB"
ACCENT_GREEN = "#16A34A"
ACCENT_AMBER = "#F59E0B"
ACCENT_RED = "#DC2626"
ACCENT_CYAN = "#0EA5E9"
ISLAND_FACE = "#E2E8F0"
ISLAND_EDGE = "#94A3B8"
FRIDGE_FACE = "#B8C2D0"
FRIDGE_EDGE = "#475569"
FRIDGE_INTERIOR = "#EAF1F8"
DOOR_FACE = "#CBD5E1"
COUCH_FACE = "#D8C3AC"
COUCH_EDGE = "#9C7C55"
ROBOT_FACE = "#2563EB"
ROBOT_EDGE = "#1E293B"
ROBOT_FRONT = "#93C5FD"
BEER_FACE = "#F59E0B"
BEER_EDGE = "#92400E"
PARTICLE_COLOR = "#22C55E"
PATH_COLOR = "#F59E0B"
TRAIL_COLOR = "#0EA5E9"

# ---------------------------------------------------------------------------
# World layout (top-down, meters-ish)
# ---------------------------------------------------------------------------
WORLD_X = (0.0, 12.4)
WORLD_Y = (0.0, 8.0)

ROOM = (0.4, 0.4, 11.6, 7.2)  # x, y, w, h

ISLAND = (4.6, 3.0, 3.0, 1.4)  # x, y, w, h

# Fridge body sits against the right wall; its front faces left (-x).
FRIDGE = (10.4, 5.1, 1.4, 2.4)  # x, y, w, h
FRIDGE_HINGE = np.array([10.4, 5.1])  # bottom-left of the front face
DOOR_LENGTH = 2.4
DOOR_MAX_DEG = 105.0
BEER_HOME = np.array([10.95, 6.35])

COUCH = (9.9, 0.8, 1.7, 1.3)
USER_CENTER = np.array([10.75, 1.45])
USER_CUP = np.array([10.2, 1.55])

ROBOT_RADIUS = 0.32
TRAY_OFFSET = 0.34

# Navigation routes (planner waypoints). The robot starts in the top-left
# corner, diagonally opposite the user in the bottom-right.
ROUTE_TO_FRIDGE = [
    (1.2, 6.9), (3.2, 6.6), (5.6, 6.4), (8.0, 6.3), (9.75, 6.35),
]
ROUTE_TO_USER = [
    (9.75, 6.35), (9.5, 4.8), (9.3, 3.4), (9.2, 2.4), (9.5, 1.95),
]

# ---------------------------------------------------------------------------
# State machine phases (name -> frame budget)
# ---------------------------------------------------------------------------
PHASES = [
    ("receive", 10),
    ("plan_to_fridge", 12),
    ("nav_to_fridge", 40),
    ("open_fridge", 12),
    ("find_beer", 10),
    ("grasp_beer", 10),
    ("close_fridge", 12),
    ("plan_to_user", 12),
    ("return_to_user", 40),
    ("deliver", 12),
    ("complete", 14),
]

STATE_LABELS = [
    "RECEIVE\nREQUEST",
    "NAVIGATE\nTO FRIDGE",
    "OPEN\nFRIDGE",
    "FIND\nBEER",
    "GRASP\nBEER",
    "CLOSE\nFRIDGE",
    "RETURN\nTO USER",
    "DELIVER\nBEER",
    "MISSION\nCOMPLETE",
]

PHASE_STATE_INDEX = {
    "receive": 0,
    "plan_to_fridge": 1,
    "nav_to_fridge": 1,
    "open_fridge": 2,
    "find_beer": 3,
    "grasp_beer": 4,
    "close_fridge": 5,
    "plan_to_user": 6,
    "return_to_user": 6,
    "deliver": 7,
    "complete": 8,
}

PHASE_CAPTION = {
    "receive": "Receiving user request",
    "plan_to_fridge": "Planning path to the refrigerator",
    "nav_to_fridge": "Navigating to the refrigerator",
    "open_fridge": "Opening the refrigerator door",
    "find_beer": "Locating the beer",
    "grasp_beer": "Grasping the beer",
    "close_fridge": "Closing the refrigerator door",
    "plan_to_user": "Planning path back to the user",
    "return_to_user": "Returning to the user",
    "deliver": "Delivering the beer",
    "complete": "Mission complete",
}

PARTICLE_COUNT = 48


@dataclass
class Scene:
    """Everything needed to render one animation frame."""

    phase: str
    state_index: int
    robot: np.ndarray
    heading: float
    door_open: float
    route: str | None
    route_reveal: float
    goal: np.ndarray | None
    beer_pos: np.ndarray | None
    beer_visible: bool
    beer_highlight: float
    particle_spread: float
    request_active: float
    caption: str
    trail: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def smoothstep(t: float) -> float:
    """Ease in/out on [0, 1]."""
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def resample_route(waypoints: list[tuple[float, float]], count: int = 260) -> np.ndarray:
    """Resample a polyline to evenly spaced points for constant-speed motion."""
    pts = np.array(waypoints, dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    samples = np.linspace(0.0, total, count)
    xs = np.interp(samples, cum, pts[:, 0])
    ys = np.interp(samples, cum, pts[:, 1])
    return np.column_stack([xs, ys])


PATH_FRIDGE = resample_route(ROUTE_TO_FRIDGE)
PATH_USER = resample_route(ROUTE_TO_USER)

ROBOT_START = PATH_FRIDGE[0].copy()
FRIDGE_STOP = PATH_FRIDGE[-1].copy()
USER_STOP = PATH_USER[-1].copy()


def pose_on_path(path: np.ndarray, frac: float) -> tuple[np.ndarray, float]:
    """Return position and heading at a fractional distance along a path."""
    frac = float(np.clip(frac, 0.0, 1.0))
    idx = frac * (len(path) - 1)
    low = int(np.floor(idx))
    high = min(low + 1, len(path) - 1)
    blend = idx - low
    pos = path[low] * (1.0 - blend) + path[high] * blend

    tangent_low = max(low - 2, 0)
    tangent_high = min(high + 2, len(path) - 1)
    tangent = path[tangent_high] - path[tangent_low]
    heading = float(np.arctan2(tangent[1], tangent[0]))
    return pos, heading


def forward_vector(heading: float) -> np.ndarray:
    return np.array([np.cos(heading), np.sin(heading)])


HEADING_TO_FRIDGE = pose_on_path(PATH_FRIDGE, 0.0)[1]
HEADING_TO_USER_PATH = pose_on_path(PATH_USER, 0.0)[1]
HEADING_FACING_FRIDGE = 0.0
_user_dir = USER_CENTER - USER_STOP
HEADING_FACING_USER = float(np.arctan2(_user_dir[1], _user_dir[0]))
_recv_dir = USER_CENTER - ROBOT_START
HEADING_RECEIVE = float(np.arctan2(_recv_dir[1], _recv_dir[0]))


# ---------------------------------------------------------------------------
# Build per-frame scene data
# ---------------------------------------------------------------------------
def build_scenes() -> list[Scene]:
    scenes: list[Scene] = []

    for phase, count in PHASES:
        for local in range(count):
            f = local / max(count - 1, 1)
            scenes.append(_build_phase_frame(phase, f))

    positions = np.array([s.robot for s in scenes])
    for i, scene in enumerate(scenes):
        scene.trail = positions[: i + 1]
    return scenes


def _build_phase_frame(phase: str, f: float) -> Scene:
    state_index = PHASE_STATE_INDEX[phase]
    caption = PHASE_CAPTION[phase]

    # Defaults.
    robot = ROBOT_START.copy()
    heading = HEADING_RECEIVE
    door_open = 0.0
    route: str | None = None
    route_reveal = 0.0
    goal: np.ndarray | None = None
    beer_pos: np.ndarray | None = BEER_HOME.copy()
    beer_visible = False
    beer_highlight = 0.0
    spread = 0.10
    request_active = 0.0

    if phase == "receive":
        robot = ROBOT_START.copy()
        heading = HEADING_RECEIVE
        spread = 0.34 - 0.18 * f
        request_active = smoothstep(min(1.0, f * 1.6))

    elif phase == "plan_to_fridge":
        robot = ROBOT_START.copy()
        heading = HEADING_TO_FRIDGE
        route = "fridge"
        route_reveal = smoothstep(f)
        goal = FRIDGE_STOP
        spread = 0.18

    elif phase == "nav_to_fridge":
        robot, heading = pose_on_path(PATH_FRIDGE, f)
        route = "fridge"
        route_reveal = 1.0
        goal = FRIDGE_STOP
        spread = 0.24

    elif phase == "open_fridge":
        robot = FRIDGE_STOP.copy()
        heading = HEADING_FACING_FRIDGE
        door_open = smoothstep(f)
        beer_visible = door_open > 0.15
        spread = 0.14

    elif phase == "find_beer":
        robot = FRIDGE_STOP.copy()
        heading = HEADING_FACING_FRIDGE
        door_open = 1.0
        beer_visible = True
        beer_highlight = 0.5 + 0.5 * np.sin(f * np.pi * 4.0)
        spread = 0.14

    elif phase == "grasp_beer":
        robot = FRIDGE_STOP.copy()
        heading = HEADING_FACING_FRIDGE
        door_open = 1.0
        tray = FRIDGE_STOP + forward_vector(HEADING_FACING_FRIDGE) * TRAY_OFFSET
        beer_pos = BEER_HOME * (1.0 - smoothstep(f)) + tray * smoothstep(f)
        beer_visible = True
        spread = 0.14

    elif phase == "close_fridge":
        robot = FRIDGE_STOP.copy()
        heading = HEADING_FACING_FRIDGE
        door_open = 1.0 - smoothstep(f)
        beer_pos = FRIDGE_STOP + forward_vector(HEADING_FACING_FRIDGE) * TRAY_OFFSET
        beer_visible = True
        spread = 0.14

    elif phase == "plan_to_user":
        robot = FRIDGE_STOP.copy()
        heading = HEADING_TO_USER_PATH
        route = "user"
        route_reveal = smoothstep(f)
        goal = USER_STOP
        beer_pos = robot + forward_vector(heading) * TRAY_OFFSET
        beer_visible = True
        spread = 0.18

    elif phase == "return_to_user":
        robot, heading = pose_on_path(PATH_USER, f)
        route = "user"
        route_reveal = 1.0
        goal = USER_STOP
        beer_pos = robot + forward_vector(heading) * TRAY_OFFSET
        beer_visible = True
        spread = 0.24

    elif phase == "deliver":
        robot = USER_STOP.copy()
        heading = HEADING_FACING_USER
        tray = USER_STOP + forward_vector(heading) * TRAY_OFFSET
        beer_pos = tray * (1.0 - smoothstep(f)) + USER_CUP * smoothstep(f)
        beer_visible = True
        spread = 0.12

    elif phase == "complete":
        robot = USER_STOP.copy()
        heading = HEADING_FACING_USER
        beer_pos = USER_CUP.copy()
        beer_visible = True
        spread = 0.11

    return Scene(
        phase=phase,
        state_index=state_index,
        robot=robot,
        heading=heading,
        door_open=door_open,
        route=route,
        route_reveal=route_reveal,
        goal=goal,
        beer_pos=beer_pos,
        beer_visible=beer_visible,
        beer_highlight=beer_highlight,
        particle_spread=spread,
        request_active=request_active,
        caption=caption,
    )


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_banner(ax: plt.Axes, current_index: int) -> None:
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    n = len(STATE_LABELS)
    slot = 100.0 / n
    for i, label in enumerate(STATE_LABELS):
        cx = (i + 0.5) * slot
        width = slot * 0.86
        if i == current_index:
            face, edge, txt, lw = "#FDE68A", ACCENT_AMBER, TEXT_COLOR, 2.0
        elif i < current_index:
            face, edge, txt, lw = "#DBEAFE", ACCENT_BLUE, ACCENT_BLUE, 1.2
        else:
            face, edge, txt, lw = "#FFFFFF", "#CBD5E1", MUTED_TEXT, 1.2

        ax.add_patch(FancyBboxPatch(
            (cx - width / 2, 20), width, 60,
            boxstyle="round,pad=0.02,rounding_size=3",
            facecolor=face, edgecolor=edge, linewidth=lw, zorder=1,
        ))
        ax.text(cx, 50, label, ha="center", va="center", fontsize=7.2,
                fontweight="bold", color=txt, zorder=2)
        if i < n - 1:
            ax.text((i + 1) * slot, 50, "\u203a", ha="center", va="center",
                    fontsize=13, color="#94A3B8", zorder=2)


def draw_fridge(ax: plt.Axes, door_open: float) -> None:
    fx, fy, fw, fh = FRIDGE

    # Body.
    ax.add_patch(FancyBboxPatch(
        (fx, fy), fw, fh, boxstyle="round,pad=0.0,rounding_size=0.08",
        facecolor=FRIDGE_FACE, edgecolor=FRIDGE_EDGE, linewidth=1.6, zorder=3,
    ))

    # Interior + shelves (only meaningful when the door is open).
    if door_open > 0.05:
        ax.add_patch(Rectangle(
            (fx + 0.06, fy + 0.08), fw - 0.12, fh - 0.16,
            facecolor=FRIDGE_INTERIOR, edgecolor="#B8C6D8", linewidth=1.0, zorder=3.2,
        ))
        for shelf_y in (fy + 0.85, fy + 1.6):
            ax.plot([fx + 0.12, fx + fw - 0.12], [shelf_y, shelf_y],
                    color="#B8C6D8", linewidth=1.2, zorder=3.3)

    # Door hinged at the bottom-left of the front face, swinging into the room.
    phi = np.deg2rad(DOOR_MAX_DEG) * float(np.clip(door_open, 0.0, 1.0))
    direction = np.array([-np.sin(phi), np.cos(phi)])
    perp = np.array([direction[1], -direction[0]])
    thickness = 0.10
    p1 = FRIDGE_HINGE
    p2 = FRIDGE_HINGE + DOOR_LENGTH * direction
    p3 = p2 + thickness * perp
    p4 = FRIDGE_HINGE + thickness * perp
    ax.add_patch(Polygon([p1, p2, p3, p4], closed=True,
                         facecolor=DOOR_FACE, edgecolor=FRIDGE_EDGE,
                         linewidth=1.5, zorder=4))
    # Handle near the free end of the door.
    handle = FRIDGE_HINGE + (DOOR_LENGTH - 0.25) * direction + 0.16 * perp
    ax.add_patch(Circle(handle, 0.07, facecolor="#64748B",
                        edgecolor=FRIDGE_EDGE, linewidth=1.0, zorder=4.1))

    ax.text(fx + fw / 2, fy + fh + 0.22, "FRIDGE", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", color=MUTED_TEXT)


def draw_beer(ax: plt.Axes, pos: np.ndarray, highlight: float) -> None:
    if highlight > 0.01:
        ax.add_patch(Circle(pos, 0.30 + 0.10 * highlight, facecolor="none",
                            edgecolor=ACCENT_RED, linewidth=1.6 + 1.4 * highlight,
                            alpha=0.4 + 0.5 * highlight, zorder=6))
    body = Rectangle((pos[0] - 0.08, pos[1] - 0.15), 0.16, 0.30,
                     facecolor=BEER_FACE, edgecolor=BEER_EDGE, linewidth=1.2, zorder=7)
    ax.add_patch(body)
    ax.add_patch(Rectangle((pos[0] - 0.06, pos[1] + 0.09), 0.12, 0.06,
                          facecolor="#FCD34D", edgecolor=BEER_EDGE, linewidth=0.8, zorder=7.1))


def draw_robot(ax: plt.Axes, pos: np.ndarray, heading: float) -> None:
    heading_deg = np.rad2deg(heading)
    ax.add_patch(Wedge(pos, ROBOT_RADIUS + 0.02, heading_deg - 42, heading_deg + 42,
                      facecolor=ROBOT_FRONT, edgecolor="none", zorder=8))
    ax.add_patch(Circle(pos, ROBOT_RADIUS, facecolor=ROBOT_FACE,
                        edgecolor=ROBOT_EDGE, linewidth=1.6, zorder=8.1))
    tip = pos + forward_vector(heading) * (ROBOT_RADIUS + 0.14)
    ax.plot([pos[0], tip[0]], [pos[1], tip[1]], color="#0F172A",
            linewidth=2.0, zorder=8.3)
    ax.add_patch(Circle(pos, 0.06, facecolor="#E2E8F0",
                        edgecolor=ROBOT_EDGE, linewidth=1.0, zorder=8.4))


def draw_scene(ax: plt.Axes, scene: Scene, rng: np.random.Generator) -> None:
    ax.clear()
    ax.set_xlim(*WORLD_X)
    ax.set_ylim(*WORLD_Y)
    ax.set_aspect("equal")
    ax.axis("off")

    # Floor + walls.
    rx, ry, rw, rh = ROOM
    ax.add_patch(FancyBboxPatch(
        (rx, ry), rw, rh, boxstyle="round,pad=0.0,rounding_size=0.12",
        facecolor=FLOOR_COLOR, edgecolor=BORDER_COLOR, linewidth=2.2, zorder=1,
    ))

    # Kitchen island (static obstacle).
    ix, iy, iw, ih = ISLAND
    ax.add_patch(FancyBboxPatch(
        (ix, iy), iw, ih, boxstyle="round,pad=0.0,rounding_size=0.12",
        facecolor=ISLAND_FACE, edgecolor=ISLAND_EDGE, linewidth=1.4, zorder=2,
    ))
    ax.text(ix + iw / 2, iy + ih / 2, "ISLAND", ha="center", va="center",
            fontsize=8, fontweight="bold", color=MUTED_TEXT, zorder=2.1)

    # Couch / user.
    cx, cy, cw, ch = COUCH
    ax.add_patch(FancyBboxPatch(
        (cx, cy), cw, ch, boxstyle="round,pad=0.0,rounding_size=0.14",
        facecolor=COUCH_FACE, edgecolor=COUCH_EDGE, linewidth=1.5, zorder=2,
    ))
    ax.text(cx + cw / 2, cy + ch + 0.22, "USER", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", color=MUTED_TEXT, zorder=2.1)

    draw_fridge(ax, scene.door_open)

    # Planned path (drawn as the planner reveals it).
    if scene.route is not None and scene.route_reveal > 0.01:
        path = PATH_FRIDGE if scene.route == "fridge" else PATH_USER
        k = max(2, int(scene.route_reveal * len(path)))
        ax.plot(path[:k, 0], path[:k, 1], color=PATH_COLOR, linewidth=2.0,
                linestyle=(0, (5, 4)), zorder=5, alpha=0.9)

    # Goal marker.
    if scene.goal is not None:
        ax.add_patch(Circle(scene.goal, 0.22, facecolor="none",
                            edgecolor=ACCENT_GREEN, linewidth=2.0, zorder=5.1))
        ax.add_patch(Circle(scene.goal, 0.07, facecolor=ACCENT_GREEN,
                            edgecolor="none", zorder=5.2))

    # Traveled trail.
    if len(scene.trail) > 2:
        ax.plot(scene.trail[:, 0], scene.trail[:, 1], color=TRAIL_COLOR,
                linewidth=1.6, alpha=0.55, zorder=4.5)

    # Localization particles + confidence ring.
    particles = rng.normal(scene.robot, scene.particle_spread, size=(PARTICLE_COUNT, 2))
    ax.add_patch(Circle(scene.robot, scene.particle_spread * 2.6, facecolor=PARTICLE_COLOR,
                        edgecolor="none", alpha=0.12, zorder=6.3))
    ax.scatter(particles[:, 0], particles[:, 1], s=32, color=PARTICLE_COLOR,
               alpha=0.85, edgecolors="#065F46", linewidths=0.4, zorder=6.5)
    ax.add_patch(Circle(scene.robot, scene.particle_spread * 2.6, facecolor="none",
                        edgecolor=PARTICLE_COLOR, linewidth=1.8, alpha=0.8, zorder=6.6))

    draw_robot(ax, scene.robot, scene.heading)

    if scene.beer_pos is not None and scene.beer_visible:
        draw_beer(ax, scene.beer_pos, scene.beer_highlight)

    # Request bubble while receiving the task.
    if scene.request_active > 0.02:
        bubble = USER_CENTER + np.array([-1.7, 1.15])
        ax.add_patch(FancyBboxPatch(
            (bubble[0] - 0.05, bubble[1] - 0.28), 2.5, 0.72,
            boxstyle="round,pad=0.06,rounding_size=0.18",
            facecolor="#FFFFFF", edgecolor=ACCENT_BLUE, linewidth=1.4,
            alpha=min(1.0, scene.request_active), zorder=9,
        ))
        ax.text(bubble[0] + 1.2, bubble[1] + 0.08, "Bring me a beer",
                ha="center", va="center", fontsize=8.5, color=TEXT_COLOR,
                alpha=min(1.0, scene.request_active), zorder=9.1)

    # Caption.
    ax.text(WORLD_X[1] / 2, ry + rh - 0.18, scene.caption, ha="center", va="top",
            fontsize=11, fontweight="bold", color=ACCENT_BLUE, zorder=9.5)
    ax.text(0.6, 0.62, "Beer-fetch mission \u00b7 top-down view", ha="left",
            va="bottom", fontsize=8.5, color=MUTED_TEXT, zorder=9.5)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def build_animation(scenes: list[Scene]) -> FuncAnimation:
    fig = plt.figure(figsize=(12.4, 8.0), dpi=100)
    fig.patch.set_facecolor(BG_COLOR)
    ax_banner = fig.add_axes([0.02, 0.87, 0.96, 0.11])
    ax_scene = fig.add_axes([0.02, 0.02, 0.96, 0.83])

    def update(frame_index: int):
        scene = scenes[frame_index]
        rng = np.random.default_rng(1234 + frame_index)
        ax_banner.clear()
        draw_banner(ax_banner, scene.state_index)
        draw_scene(ax_scene, scene, rng)
        return []

    return FuncAnimation(fig, update, frames=len(scenes),
                         interval=1000 / FPS, blit=False, repeat=True)


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    scenes = build_scenes()
    anim = build_animation(scenes)
    output_path = OUTPUT_DIRECTORY / GIF_NAME
    anim.save(str(output_path), writer=PillowWriter(fps=FPS), dpi=100)
    plt.close("all")
    print(f"Saved animation to {output_path} ({len(scenes)} frames)")


if __name__ == "__main__":
    main()
