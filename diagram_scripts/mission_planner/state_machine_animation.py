#!/usr/bin/env python3
"""Create a slide-friendly GIF visualizing a ROS navigation state machine loop.

The animation highlights the execution path through a simplified
NavigateToFridgeState.execute() flow and emphasizes the repeated loop where
navigation remains pending while the state machine keeps spinning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "state_machine_animation_outputs"
FRAME_PREFIX = "slide_frame"

BG_COLOR = "#F8FAFC"
CARD_COLOR = "#FFFFFF"
BORDER_COLOR = "#334155"
TEXT_COLOR = "#0F172A"
MUTED_TEXT = "#475569"
ACCENT_BLUE = "#2563EB"
ACCENT_GREEN = "#16A34A"
ACCENT_AMBER = "#F59E0B"
ACCENT_RED = "#DC2626"
ACCENT_CYAN = "#0EA5E9"
HIGHLIGHT = "#DBEAFE"


@dataclass
class StateSnapshot:
    """Inputs and persistent futures visible during one execute() tick."""

    fridge_pose: str
    send_future: str
    goal_handle: str
    result_future: str


@dataclass
class FrameSpec:
    title: str
    description: str
    line_idx: int
    tick: str
    executed: tuple[int, ...]
    snapshot: StateSnapshot
    status: str
    next_state: str


CODE_LINES = [
    "class NavigateToFridgeState:",
    "    def __init__(self, node):",
    "        self.nav = ActionClient(node, NavigateToPose, 'navigate_to_pose')",
    "        self.send_future = None",
    "        self.goal_handle = None",
    "        self.result_future = None",
    "    def execute(self, fridge_pose):",
    "        if self.send_future is None:",
    "            if not self.nav.server_is_ready(): return State.NAVIGATE_TO_FRIDGE",
    "            goal = NavigateToPose.Goal(); goal.pose = fridge_pose",
    "            self.send_future = self.nav.send_goal_async(goal)",
    "            return State.NAVIGATE_TO_FRIDGE",
    "        if self.goal_handle is None:",
    "            if not self.send_future.done(): return State.NAVIGATE_TO_FRIDGE",
    "            self.goal_handle = self.send_future.result()",
    "            if not self.goal_handle.accepted: self.reset(); return State.RECOVERY",
    "            self.result_future = self.goal_handle.get_result_async()",
    "            return State.NAVIGATE_TO_FRIDGE",
    "        if not self.result_future.done(): return State.NAVIGATE_TO_FRIDGE",
    "        status = self.result_future.result().status",
    "        self.reset()",
    "        if status == GoalStatus.STATUS_SUCCEEDED: return State.OPEN_FRIDGE",
    "        return State.RECOVERY",
    "    def reset(self):",
    "        self.send_future = None; self.goal_handle = None; self.result_future = None",
]

LINE_INDEX = {name: idx for idx, name in enumerate([
    "class", "def_init", "init_client", "init_send", "init_goal", "init_result",
    "def_execute", "if_send_none", "if_server", "build_goal", "send_async",
    "return_a", "if_handle_none", "if_send_done", "get_handle", "if_rejected",
    "get_result", "return_b", "if_result_done", "read_status", "reset_call",
    "if_success", "return_recovery", "def_reset", "reset_body",
])}

POSE = "PoseStamped(x=2.40, y=4.10)"
NONE = "None"
SEND_PENDING = "<Future pending>"
SEND_DONE = "<Future done>"
HANDLE_OK = "GoalHandle(accepted=True)"
HANDLE_REJECTED = "GoalHandle(accepted=False)"
RESULT_PENDING = "<Future pending>"
RESULT_DONE = "<Future done>"


def build_frames() -> list[FrameSpec]:
    """Walk through the state machine as it is polled tick after tick.

    execute() never blocks: each call advances one of three phases (send the
    goal, wait for acceptance, wait for the result) and returns a State. The
    three instance futures act as memory that routes each new tick to the
    correct phase.
    """

    frames: list[FrameSpec] = []
    executed: list[int] = []

    def add(
        title: str,
        description: str,
        line_key: str,
        tick: str,
        snapshot: StateSnapshot,
        status: str,
        next_state: str,
    ) -> None:
        line_idx = LINE_INDEX[line_key]
        if line_idx not in executed:
            executed.append(line_idx)
        frames.append(FrameSpec(
            title=title,
            description=description,
            line_idx=line_idx,
            tick=tick,
            executed=tuple(executed),
            snapshot=snapshot,
            status=status,
            next_state=next_state,
        ))

    # -- Construction --------------------------------------------------------
    add(
        "Persistent state",
        "execute() is polled every tick. Three futures remember progress; they all start as None.",
        "init_send", "constructor",
        StateSnapshot(POSE, NONE, NONE, NONE),
        "send/goal/result futures set to None", "(not returned yet)",
    )

    # -- Phase 1 (tick 1): send the goal ------------------------------------
    add(
        "send-goal phase",
        "No goal has been sent, so this guard is True and we enter the send phase.",
        "if_send_none", "tick 1",
        StateSnapshot(POSE, NONE, NONE, NONE),
        "self.send_future is None -> True", "NAVIGATE_TO_FRIDGE",
    )
    add(
        "Check the server",
        "Only send once the Nav2 action server is ready.",
        "if_server", "tick 1",
        StateSnapshot(POSE, NONE, NONE, NONE),
        "server_is_ready() -> True", "NAVIGATE_TO_FRIDGE",
    )
    add(
        "Build the goal",
        "Wrap the target fridge pose in a NavigateToPose goal.",
        "build_goal", "tick 1",
        StateSnapshot(POSE, NONE, NONE, NONE),
        "goal.pose = fridge_pose", "NAVIGATE_TO_FRIDGE",
    )
    add(
        "Send asynchronously",
        "Send without blocking and keep the returned future.",
        "send_async", "tick 1",
        StateSnapshot(POSE, SEND_PENDING, NONE, NONE),
        "send_future = send_goal_async(goal)", "NAVIGATE_TO_FRIDGE",
    )
    add(
        "Stay in the state",
        "Return NAVIGATE_TO_FRIDGE so execute() runs again next tick.",
        "return_a", "tick 1",
        StateSnapshot(POSE, SEND_PENDING, NONE, NONE),
        "returned NAVIGATE_TO_FRIDGE", "NAVIGATE_TO_FRIDGE",
    )

    # -- Phase 2 (tick 2+): wait for the goal to be accepted ----------------
    add(
        "guard skipped",
        "send_future now exists, so the send-goal branch is skipped.",
        "if_send_none", "tick 2",
        StateSnapshot(POSE, SEND_PENDING, NONE, NONE),
        "self.send_future is None -> False", "NAVIGATE_TO_FRIDGE",
    )
    add(
        "Accept-wait phase",
        "goal_handle is still None, so enter the acceptance branch.",
        "if_handle_none", "tick 2",
        StateSnapshot(POSE, SEND_PENDING, NONE, NONE),
        "self.goal_handle is None -> True", "NAVIGATE_TO_FRIDGE",
    )
    add(
        "Poll for a reply",
        "If Nav2 has not answered yet, return and try again next tick.",
        "if_send_done", "tick 2",
        StateSnapshot(POSE, SEND_PENDING, NONE, NONE),
        "send_future.done() -> False", "NAVIGATE_TO_FRIDGE",
    )
    add(
        "reply ready",
        "Eventually the server answers and send_future is done.",
        "if_send_done", "tick 3",
        StateSnapshot(POSE, SEND_DONE, NONE, NONE),
        "send_future.done() -> True", "NAVIGATE_TO_FRIDGE",
    )
    add(
        "Read the goal handle",
        "Unwrap the handle describing the accepted/rejected goal.",
        "get_handle", "tick 3",
        StateSnapshot(POSE, SEND_DONE, HANDLE_OK, NONE),
        "goal_handle = send_future.result()", "NAVIGATE_TO_FRIDGE",
    )
    add(
        "Was it accepted?",
        "A rejected goal would reset and go to RECOVERY; here it is accepted.",
        "if_rejected", "tick 3",
        StateSnapshot(POSE, SEND_DONE, HANDLE_OK, NONE),
        "goal_handle.accepted -> True", "NAVIGATE_TO_FRIDGE",
    )
    add(
        "Request the result",
        "Ask Nav2 for the final result as another async future.",
        "get_result", "tick 3",
        StateSnapshot(POSE, SEND_DONE, HANDLE_OK, RESULT_PENDING),
        "result_future = get_result_async()", "NAVIGATE_TO_FRIDGE",
    )
    add(
        "Stay in the state",
        "Return NAVIGATE_TO_FRIDGE again while the robot drives.",
        "return_b", "tick 3",
        StateSnapshot(POSE, SEND_DONE, HANDLE_OK, RESULT_PENDING),
        "returned NAVIGATE_TO_FRIDGE", "NAVIGATE_TO_FRIDGE",
    )

    # -- Phase 3 (tick 4+): wait for navigation to finish -------------------
    add(
        "both guards skipped",
        "send_future and goal_handle are set, so both early branches are skipped.",
        "if_result_done", "tick 4",
        StateSnapshot(POSE, SEND_DONE, HANDLE_OK, RESULT_PENDING),
        "result_future.done() -> False", "NAVIGATE_TO_FRIDGE",
    )
    add(
        "navigation done",
        "When the robot arrives (or fails) result_future becomes done.",
        "if_result_done", "tick N",
        StateSnapshot(POSE, SEND_DONE, HANDLE_OK, RESULT_DONE),
        "result_future.done() -> True", "NAVIGATE_TO_FRIDGE",
    )
    add(
        "Read the status",
        "Extract the terminal GoalStatus from the result.",
        "read_status", "tick N",
        StateSnapshot(POSE, SEND_DONE, HANDLE_OK, RESULT_DONE),
        "status = result.status", "NAVIGATE_TO_FRIDGE",
    )
    add(
        "Reset for next time",
        "Clear all three futures so the state can run fresh later.",
        "reset_call", "tick N",
        StateSnapshot(POSE, NONE, NONE, NONE),
        "reset() -> futures cleared", "NAVIGATE_TO_FRIDGE",
    )
    add(
        "Success -> OPEN_FRIDGE",
        "On STATUS_SUCCEEDED the state transitions out to open the fridge.",
        "if_success", "tick N",
        StateSnapshot(POSE, NONE, NONE, NONE),
        "status == SUCCEEDED -> True", "OPEN_FRIDGE",
    )

    # -- Alternate outcome A: goal rejected ---------------------------------
    rejected_path = (
        LINE_INDEX["init_send"], LINE_INDEX["if_send_none"], LINE_INDEX["if_server"],
        LINE_INDEX["build_goal"], LINE_INDEX["send_async"], LINE_INDEX["return_a"],
        LINE_INDEX["if_handle_none"], LINE_INDEX["if_send_done"], LINE_INDEX["get_handle"],
        LINE_INDEX["if_rejected"],
    )
    frames.append(FrameSpec(
        title="Alternate · goal rejected",
        description="If Nav2 rejects the goal, reset the futures and go to RECOVERY.",
        line_idx=LINE_INDEX["if_rejected"],
        tick="tick 3",
        executed=rejected_path,
        snapshot=StateSnapshot(POSE, SEND_DONE, HANDLE_REJECTED, NONE),
        status="goal_handle.accepted -> False; reset()",
        next_state="RECOVERY",
    ))

    # -- Alternate outcome B: navigation failed -----------------------------
    failed_path = rejected_path[:-1] + (
        LINE_INDEX["get_result"], LINE_INDEX["return_b"], LINE_INDEX["if_result_done"],
        LINE_INDEX["read_status"], LINE_INDEX["reset_call"], LINE_INDEX["if_success"],
        LINE_INDEX["return_recovery"],
    )
    frames.append(FrameSpec(
        title="Alternate · navigation failed",
        description="Any non-success status falls through to RECOVERY after reset.",
        line_idx=LINE_INDEX["return_recovery"],
        tick="tick N",
        executed=failed_path,
        snapshot=StateSnapshot(POSE, NONE, NONE, NONE),
        status="status != SUCCEEDED -> RECOVERY",
        next_state="RECOVERY",
    ))

    return frames


def draw_code_panel(ax: plt.Axes, frame: FrameSpec) -> None:
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.add_patch(FancyBboxPatch(
        (2, 2), 96, 96,
        boxstyle="round,pad=0.02,rounding_size=2",
        linewidth=1.2,
        edgecolor=BORDER_COLOR,
        facecolor=CARD_COLOR,
        zorder=0,
    ))

    ax.text(5, 92, "NavigateToFridgeState", fontsize=17, fontweight="bold", color=TEXT_COLOR)
    ax.text(5, 87, "Non-blocking state polled once per tick", fontsize=9.4, color=MUTED_TEXT)

    current_idx = frame.line_idx
    executed = set(frame.executed)
    line_height = 3.8
    start_y = 81
    for idx, line in enumerate(CODE_LINES):
        y = start_y - idx * line_height
        if y < 8:
            continue
        if idx == current_idx:
            ax.add_patch(Rectangle((3.4, y - 2.4), 88.0, 3.0, facecolor=HIGHLIGHT, edgecolor=ACCENT_BLUE, linewidth=1.2, zorder=1))
            ax.text(2.5, y - 0.6, "▶", fontsize=11, color=ACCENT_BLUE, ha="center", va="center", zorder=2)
            ax.text(5.2, y - 0.6, line, fontsize=8.6, color=ACCENT_BLUE, fontfamily="DejaVu Sans Mono", zorder=2)
        elif idx in executed:
            ax.text(5.2, y - 0.6, line, fontsize=8.6, color=ACCENT_BLUE, fontfamily="DejaVu Sans Mono", zorder=2)
        else:
            ax.text(5.2, y - 0.6, line, fontsize=8.6, color=MUTED_TEXT, fontfamily="DejaVu Sans Mono", zorder=2)


def draw_state_panel(ax: plt.Axes, frame: FrameSpec) -> None:
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.add_patch(FancyBboxPatch(
        (2, 2), 96, 96,
        boxstyle="round,pad=0.02,rounding_size=2",
        linewidth=1.2,
        edgecolor=BORDER_COLOR,
        facecolor=CARD_COLOR,
        zorder=0,
    ))

    ax.text(5, 93, "State variables and this tick", fontsize=17, fontweight="bold", color=TEXT_COLOR)
    ax.text(5, 88, f"{frame.tick} · {frame.title}", fontsize=10.2, color=ACCENT_BLUE, fontweight="bold")
    ax.text(5, 84, frame.description, fontsize=9.0, color=MUTED_TEXT, wrap=True)

    snap = frame.snapshot

    # ---- Input -------------------------------------------------------------
    ax.text(5, 74, "Input", fontsize=11.5, fontweight="bold", color=TEXT_COLOR)
    ax.text(5, 69, "# Target pose handed to execute() each tick", fontsize=8.2, fontfamily="DejaVu Sans Mono", color=ACCENT_CYAN)
    ax.text(5, 64.5, f"fridge_pose = {snap.fridge_pose}", fontsize=8.8, fontfamily="DejaVu Sans Mono", color=TEXT_COLOR)

    # ---- Persistent futures -----------------------------------------------
    ax.text(5, 54, "Persistent state", fontsize=11.5, fontweight="bold", color=TEXT_COLOR)
    ax.text(5, 49, "# Futures kept between calls; they route each tick", fontsize=8.2, fontfamily="DejaVu Sans Mono", color=ACCENT_CYAN)
    ax.text(5, 44.5, f"self.send_future   = {snap.send_future}", fontsize=8.8, fontfamily="DejaVu Sans Mono", color=TEXT_COLOR)
    ax.text(5, 40, f"self.goal_handle   = {snap.goal_handle}", fontsize=8.8, fontfamily="DejaVu Sans Mono", color=TEXT_COLOR)
    ax.text(5, 35.5, f"self.result_future = {snap.result_future}", fontsize=8.8, fontfamily="DejaVu Sans Mono", color=TEXT_COLOR)

    # ---- Outcome of the highlighted line ----------------------------------
    ax.text(5, 25, "This tick", fontsize=11.5, fontweight="bold", color=TEXT_COLOR)
    ax.text(5, 20, "# What the highlighted line does and returns", fontsize=8.2, fontfamily="DejaVu Sans Mono", color=ACCENT_CYAN)
    ax.text(5, 15.5, f"status = {frame.status}", fontsize=8.8, fontfamily="DejaVu Sans Mono", color=TEXT_COLOR)
    ax.text(5, 11, f"returns -> {frame.next_state}", fontsize=8.8, fontfamily="DejaVu Sans Mono", color=ACCENT_GREEN, fontweight="bold")


def render_frame(frame: FrameSpec, output_path: Path) -> None:
    fig = plt.figure(figsize=(16, 8.2), dpi=140)
    fig.patch.set_facecolor(BG_COLOR)
    ax_code = fig.add_axes([0.04, 0.08, 0.56, 0.84])
    ax_state = fig.add_axes([0.62, 0.08, 0.34, 0.84])

    draw_code_panel(ax_code, frame)
    draw_state_panel(ax_state, frame)

    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    frames = build_frames()

    for idx, frame in enumerate(frames, start=1):
        output_path = OUTPUT_DIRECTORY / f"{FRAME_PREFIX}_{idx:03d}.png"
        render_frame(frame, output_path)
        print(f"Saved frame to {output_path}")


if __name__ == "__main__":
    main()
