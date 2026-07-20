"""Generate a small floor-plan occupancy grid for an A* lecture.

Obstacles are black, free cells are white, the robot is a red circle, and the
goal is a blue star. The shortest path found by A* is drawn as a green line.
"""

import heapq
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Map settings
# ---------------------------------------------------------------------------
GRID_SIZE = 20

# Coordinates use the form (column, row), with (0, 0) at the lower-left.
ROBOT_POSITION = (3, 4)
GOAL_POSITION = (16, 15)

# Output files are written next to this script.
OUTPUT_DIRECTORY = Path(__file__).resolve().parent


def create_floor_plan(size: int = GRID_SIZE) -> np.ndarray:
    """Return an occupancy grid containing several rooms and obstacles.

    Cell values:
        0 = free space
        1 = occupied space
    """
    grid = np.zeros((size, size), dtype=np.uint8)

    # Enclose the map with one-cell-thick walls.
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1

    # Main partition: the doorway is deliberately below the direct line
    # between the robot and the goal, forcing a visible detour.
    main_wall_column = 9
    grid[1:-1, main_wall_column] = 1
    grid[5:8, main_wall_column] = 0

    # Divide the right side into lower and upper rooms. Its doorway is offset
    # toward the right side, creating a second turn in a future planned path.
    upper_room_wall_row = 11
    grid[upper_room_wall_row, main_wall_column:-1] = 1
    grid[upper_room_wall_row, 15:17] = 0

    # Add an alcove in the left room. The opening faces toward the center of
    # the map, so it looks like part of a floor plan rather than random noise.
    grid[12, 1:7] = 1
    grid[12:16, 6] = 1
    grid[12, 4:6] = 0

    # Add two compact obstacles that can later demonstrate how A* navigates
    # around furniture, cabinets, or other occupied regions.
    grid[2:4, 13:16] = 1
    grid[15:17, 11:13] = 1

    return grid


def verify_marker_position(
    grid: np.ndarray, position: tuple[int, int], marker_name: str
) -> None:
    """Raise an error if a marker is outside the map or lies on a wall."""
    column, row = position
    rows, columns = grid.shape

    if not (0 <= column < columns and 0 <= row < rows):
        raise ValueError(f"{marker_name} position {position} is outside the map.")

    if grid[row, column] == 1:
        raise ValueError(f"{marker_name} position {position} lies on a wall.")


def verify_direct_route_is_blocked(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> None:
    """Confirm that the straight segment from start to goal crosses a wall."""
    start_column, start_row = start
    goal_column, goal_row = goal

    sample_count = 500
    sampled_columns = np.rint(
        np.linspace(start_column, goal_column, sample_count)
    ).astype(int)
    sampled_rows = np.rint(np.linspace(start_row, goal_row, sample_count)).astype(int)

    if not np.any(grid[sampled_rows, sampled_columns] == 1):
        raise ValueError("The robot still has an unobstructed direct route to the goal.")


def find_shortest_path(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    """Return the shortest free-space path from start to goal using A*.

    Positions use the form (column, row). Movement is allowed in the four
    cardinal directions and the four diagonals. The heuristic is the octile
    distance, which is admissible for eight-connected grids.
    """
    rows, columns = grid.shape

    def is_free(column: int, row: int) -> bool:
        return 0 <= column < columns and 0 <= row < rows and grid[row, column] == 0

    straight_cost = 1.0
    diagonal_cost = np.sqrt(2.0)
    neighbor_offsets = (
        (1, 0, straight_cost),
        (-1, 0, straight_cost),
        (0, 1, straight_cost),
        (0, -1, straight_cost),
        (1, 1, diagonal_cost),
        (1, -1, diagonal_cost),
        (-1, 1, diagonal_cost),
        (-1, -1, diagonal_cost),
    )

    def heuristic(node: tuple[int, int]) -> float:
        delta_column = abs(node[0] - goal[0])
        delta_row = abs(node[1] - goal[1])
        larger, smaller = max(delta_column, delta_row), min(delta_column, delta_row)
        return straight_cost * (larger - smaller) + diagonal_cost * smaller

    open_heap: list[tuple[float, tuple[int, int]]] = [(heuristic(start), start)]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    cost_so_far: dict[tuple[int, int], float] = {start: 0.0}

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        current_column, current_row = current
        for column_step, row_step, step_cost in neighbor_offsets:
            neighbor = (current_column + column_step, current_row + row_step)
            if not is_free(*neighbor):
                continue

            # Prevent diagonal moves that would clip the corner of a wall.
            if column_step != 0 and row_step != 0:
                if not is_free(current_column + column_step, current_row) or not is_free(
                    current_column, current_row + row_step
                ):
                    continue

            new_cost = cost_so_far[current] + step_cost
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor)
                heapq.heappush(open_heap, (priority, neighbor))
                came_from[neighbor] = current

    raise ValueError("No path exists between the robot and the goal.")


def plot_occupancy_grid(
    grid: np.ndarray,
    robot_position: tuple[int, int],
    goal_position: tuple[int, int],
) -> None:
    """Plot the occupancy grid and save a slide-ready PNG file."""
    verify_marker_position(grid, robot_position, "Robot")
    verify_marker_position(grid, goal_position, "Goal")
    verify_direct_route_is_blocked(grid, robot_position, goal_position)

    path = find_shortest_path(grid, robot_position, goal_position)

    rows, columns = grid.shape
    figure, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)

    occupancy_colors = ListedColormap(["#FFFFFF", "#252525"])
    axis.imshow(
        grid,
        cmap=occupancy_colors,
        origin="lower",
        vmin=0,
        vmax=1,
        interpolation="none",
        extent=(-0.5, columns - 0.5, -0.5, rows - 0.5),
    )

    # Draw one grid line around every occupancy-grid cell.
    axis.set_xticks(np.arange(-0.5, columns, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    axis.grid(which="minor", color="#B8B8B8", linewidth=0.55)
    axis.tick_params(which="minor", bottom=False, left=False)

    # Label only every second cell so the axes remain readable on a slide.
    major_ticks = np.arange(0, columns, 2)
    axis.set_xticks(major_ticks)
    axis.set_yticks(major_ticks)
    axis.tick_params(axis="both", labelsize=10, length=0, pad=6)

    robot_column, robot_row = robot_position
    goal_column, goal_row = goal_position

    path_columns = [column for column, _ in path]
    path_rows = [row for _, row in path]
    axis.plot(
        path_columns,
        path_rows,
        color="#2E7D32",
        linewidth=3.5,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=2,
    )

    axis.scatter(
        robot_column,
        robot_row,
        s=650,
        marker="o",
        facecolor="#E53935",
        edgecolor="white",
        linewidth=2.5,
        zorder=3,
    )
    axis.scatter(
        goal_column,
        goal_row,
        s=850,
        marker="*",
        facecolor="#1976D2",
        edgecolor="white",
        linewidth=1.8,
        zorder=3,
    )

    axis.set_title(
        "A* Shortest Path on an Occupancy Grid", fontsize=20, weight="bold", pad=14
    )
    axis.set_xlabel("Grid column, x", fontsize=13, labelpad=10)
    axis.set_ylabel("Grid row, y", fontsize=13, labelpad=10)
    axis.set_xlim(-0.5, columns - 0.5)
    axis.set_ylim(-0.5, rows - 0.5)
    axis.set_aspect("equal")

    legend_items = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#E53935",
            markeredgecolor="white",
            markersize=13,
            label="Robot",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="none",
            markerfacecolor="#1976D2",
            markeredgecolor="white",
            markersize=17,
            label="Goal",
        ),
        Line2D(
            [0],
            [0],
            color="#2E7D32",
            linewidth=3.5,
            label="A* path",
        ),
    ]
    axis.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=3,
        frameon=False,
        fontsize=12,
    )

    png_path = OUTPUT_DIRECTORY / "a_star_two_room_map.png"
    figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    print(f"Saved {png_path}")


def main() -> None:
    occupancy_grid = create_floor_plan()
    plot_occupancy_grid(occupancy_grid, ROBOT_POSITION, GOAL_POSITION)


if __name__ == "__main__":
    main()
