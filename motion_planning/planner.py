from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
import random
from time import perf_counter
from typing import Iterable

from motion_planning.map import GridMap, Point


ASTAR = "astar"
DIJKSTRA = "dijkstra"
GREEDY = "greedy"
RRT = "rrt"
BFS = "bfs"
WEIGHTED_ASTAR = "weighted_astar"
BUG2 = "bug2"
POTENTIAL_FIELDS = "potential_fields"
WEIGHTED_ASTAR_WEIGHT = 1.8


@dataclass
class PathResult:
    algorithm: str
    path: list[Point] | None
    expanded_nodes: int
    total_cost: float | None
    elapsed_ms: float = 0.0

    @property
    def path_length(self) -> int | None:
        if self.path is None:
            return None
        return max(0, len(self.path) - 1)


def manhattan_distance(a: Point, b: Point) -> int:
    return abs(a.row - b.row) + abs(a.col - b.col)


def find_path(
    algorithm: str,
    grid_map: GridMap,
    start: Point,
    goal: Point,
    blocked_points: Iterable[Point] | None = None,
    obstacle_points: Iterable[Point] | None = None,
    risk_radius: int = 0,
    risk_weight: float = 0.0,
) -> PathResult:
    start_time = perf_counter()
    if algorithm == ASTAR:
        result = astar_path(
            grid_map,
            start,
            goal,
            blocked_points,
            obstacle_points=obstacle_points,
            risk_radius=risk_radius,
            risk_weight=risk_weight,
        )
    elif algorithm == DIJKSTRA:
        result = dijkstra_path(
            grid_map,
            start,
            goal,
            blocked_points,
            obstacle_points=obstacle_points,
            risk_radius=risk_radius,
            risk_weight=risk_weight,
        )
    elif algorithm == GREEDY:
        result = greedy_best_first_path(
            grid_map,
            start,
            goal,
            blocked_points,
            obstacle_points=obstacle_points,
            risk_radius=risk_radius,
            risk_weight=risk_weight,
        )
    elif algorithm == RRT:
        result = rrt_path(
            grid_map,
            start,
            goal,
            blocked_points,
            obstacle_points=obstacle_points,
            risk_radius=risk_radius,
            risk_weight=risk_weight,
        )
    elif algorithm == BFS:
        result = bfs_path(
            grid_map,
            start,
            goal,
            blocked_points,
            obstacle_points=obstacle_points,
            risk_radius=risk_radius,
            risk_weight=risk_weight,
        )
    elif algorithm == WEIGHTED_ASTAR:
        result = weighted_astar_path(
            grid_map,
            start,
            goal,
            blocked_points,
            obstacle_points=obstacle_points,
            risk_radius=risk_radius,
            risk_weight=risk_weight,
        )
    elif algorithm == BUG2:
        result = bug2_path(
            grid_map,
            start,
            goal,
            blocked_points,
            obstacle_points=obstacle_points,
            risk_radius=risk_radius,
            risk_weight=risk_weight,
        )
    elif algorithm == POTENTIAL_FIELDS:
        result = potential_fields_path(
            grid_map,
            start,
            goal,
            blocked_points,
            obstacle_points=obstacle_points,
            risk_radius=risk_radius,
            risk_weight=risk_weight,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    result.elapsed_ms = (perf_counter() - start_time) * 1000
    return result


def bfs_path(
    grid_map: GridMap,
    start: Point,
    goal: Point,
    blocked_points: Iterable[Point] | None = None,
    obstacle_points: Iterable[Point] | None = None,
    risk_radius: int = 0,
    risk_weight: float = 0.0,
) -> PathResult:
    blocked = _sanitize_blocked_points(start, goal, blocked_points)
    obstacles = set(obstacle_points or [])
    frontier = deque([start])
    came_from: dict[Point, Point | None] = {start: None}
    expanded_nodes = 0

    while frontier:
        current = frontier.popleft()
        expanded_nodes += 1
        if current == goal:
            path = _reconstruct_path(came_from, goal)
            return PathResult(
                BFS,
                path,
                expanded_nodes,
                path_cost(path, obstacle_points=obstacles, risk_radius=risk_radius, risk_weight=risk_weight),
            )

        for neighbor in grid_map.neighbors(current):
            if neighbor in blocked or neighbor in came_from:
                continue
            came_from[neighbor] = current
            frontier.append(neighbor)

    return PathResult(BFS, None, expanded_nodes, None)


def weighted_astar_path(
    grid_map: GridMap,
    start: Point,
    goal: Point,
    blocked_points: Iterable[Point] | None = None,
    obstacle_points: Iterable[Point] | None = None,
    risk_radius: int = 0,
    risk_weight: float = 0.0,
) -> PathResult:
    blocked = _sanitize_blocked_points(start, goal, blocked_points)
    obstacles = set(obstacle_points or [])

    frontier: list[tuple[float, int, Point]] = []
    tie_breaker = count()
    heappush(frontier, (0.0, next(tie_breaker), start))

    came_from: dict[Point, Point | None] = {start: None}
    cost_so_far: dict[Point, float] = {start: 0.0}
    expanded_nodes = 0

    while frontier:
        _, _, current = heappop(frontier)
        expanded_nodes += 1
        if current == goal:
            return PathResult(
                WEIGHTED_ASTAR,
                _reconstruct_path(came_from, goal),
                expanded_nodes,
                cost_so_far[goal],
            )

        for neighbor in grid_map.neighbors(current):
            if neighbor in blocked:
                continue

            new_cost = cost_so_far[current] + movement_cost(
                neighbor,
                obstacle_points=obstacles,
                risk_radius=risk_radius,
                risk_weight=risk_weight,
            )
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + WEIGHTED_ASTAR_WEIGHT * manhattan_distance(neighbor, goal)
                heappush(frontier, (priority, next(tie_breaker), neighbor))
                came_from[neighbor] = current

    return PathResult(WEIGHTED_ASTAR, None, expanded_nodes, None)


def astar_path(
    grid_map: GridMap,
    start: Point,
    goal: Point,
    blocked_points: Iterable[Point] | None = None,
    obstacle_points: Iterable[Point] | None = None,
    risk_radius: int = 0,
    risk_weight: float = 0.0,
) -> PathResult:
    blocked = _sanitize_blocked_points(start, goal, blocked_points)
    obstacles = set(obstacle_points or [])

    frontier: list[tuple[float, int, Point]] = []
    tie_breaker = count()
    heappush(frontier, (0.0, next(tie_breaker), start))

    came_from: dict[Point, Point | None] = {start: None}
    cost_so_far: dict[Point, float] = {start: 0.0}
    expanded_nodes = 0

    while frontier:
        _, _, current = heappop(frontier)
        expanded_nodes += 1
        if current == goal:
            return PathResult(ASTAR, _reconstruct_path(came_from, goal), expanded_nodes, cost_so_far[goal])

        for neighbor in grid_map.neighbors(current):
            if neighbor in blocked:
                continue

            new_cost = cost_so_far[current] + movement_cost(
                neighbor,
                obstacle_points=obstacles,
                risk_radius=risk_radius,
                risk_weight=risk_weight,
            )
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + manhattan_distance(neighbor, goal)
                heappush(frontier, (priority, next(tie_breaker), neighbor))
                came_from[neighbor] = current

    return PathResult(ASTAR, None, expanded_nodes, None)


def dijkstra_path(
    grid_map: GridMap,
    start: Point,
    goal: Point,
    blocked_points: Iterable[Point] | None = None,
    obstacle_points: Iterable[Point] | None = None,
    risk_radius: int = 0,
    risk_weight: float = 0.0,
) -> PathResult:
    blocked = _sanitize_blocked_points(start, goal, blocked_points)
    obstacles = set(obstacle_points or [])

    frontier: list[tuple[float, int, Point]] = []
    tie_breaker = count()
    heappush(frontier, (0.0, next(tie_breaker), start))

    came_from: dict[Point, Point | None] = {start: None}
    cost_so_far: dict[Point, float] = {start: 0.0}
    expanded_nodes = 0

    while frontier:
        _, _, current = heappop(frontier)
        expanded_nodes += 1
        if current == goal:
            return PathResult(DIJKSTRA, _reconstruct_path(came_from, goal), expanded_nodes, cost_so_far[goal])

        for neighbor in grid_map.neighbors(current):
            if neighbor in blocked:
                continue

            new_cost = cost_so_far[current] + movement_cost(
                neighbor,
                obstacle_points=obstacles,
                risk_radius=risk_radius,
                risk_weight=risk_weight,
            )
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                heappush(frontier, (new_cost, next(tie_breaker), neighbor))
                came_from[neighbor] = current

    return PathResult(DIJKSTRA, None, expanded_nodes, None)


def greedy_best_first_path(
    grid_map: GridMap,
    start: Point,
    goal: Point,
    blocked_points: Iterable[Point] | None = None,
    obstacle_points: Iterable[Point] | None = None,
    risk_radius: int = 0,
    risk_weight: float = 0.0,
) -> PathResult:
    blocked = _sanitize_blocked_points(start, goal, blocked_points)
    obstacles = set(obstacle_points or [])

    frontier: list[tuple[int, int, Point]] = []
    tie_breaker = count()
    heappush(frontier, (manhattan_distance(start, goal), next(tie_breaker), start))

    came_from: dict[Point, Point | None] = {start: None}
    expanded_nodes = 0

    while frontier:
        _, _, current = heappop(frontier)
        expanded_nodes += 1
        if current == goal:
            path = _reconstruct_path(came_from, goal)
            return PathResult(
                GREEDY,
                path,
                expanded_nodes,
                path_cost(path, obstacle_points=obstacles, risk_radius=risk_radius, risk_weight=risk_weight),
            )

        for neighbor in grid_map.neighbors(current):
            if neighbor in blocked or neighbor in came_from:
                continue
            came_from[neighbor] = current
            heappush(frontier, (manhattan_distance(neighbor, goal), next(tie_breaker), neighbor))

    return PathResult(GREEDY, None, expanded_nodes, None)


def rrt_path(
    grid_map: GridMap,
    start: Point,
    goal: Point,
    blocked_points: Iterable[Point] | None = None,
    obstacle_points: Iterable[Point] | None = None,
    risk_radius: int = 0,
    risk_weight: float = 0.0,
) -> PathResult:
    blocked = _sanitize_blocked_points(start, goal, blocked_points)
    obstacles = set(obstacle_points or [])

    free_points = [
        Point(row, col)
        for row in range(grid_map.rows)
        for col in range(grid_map.cols)
        if grid_map.is_free(Point(row, col)) and Point(row, col) not in blocked
    ]
    if goal not in free_points:
        free_points.append(goal)

    seed = (
        start.row * 10007
        + start.col * 1009
        + goal.row * 313
        + goal.col * 37
        + len(blocked) * 17
        + len(obstacles) * 23
    )
    rng = random.Random(seed)

    tree_nodes = [start]
    parents: dict[Point, Point | None] = {start: None}
    expanded_nodes = 0
    max_iterations = max(200, grid_map.rows * grid_map.cols * 6)

    for _ in range(max_iterations):
        sample = goal if rng.random() < 0.25 else rng.choice(free_points)
        nearest = min(tree_nodes, key=lambda node: manhattan_distance(node, sample))
        next_point = _step_towards_sample(grid_map, nearest, sample, blocked)
        expanded_nodes += 1

        if next_point is None or next_point in parents:
            continue

        parents[next_point] = nearest
        tree_nodes.append(next_point)

        if next_point == goal:
            path = _reconstruct_path(parents, goal)
            return PathResult(
                RRT,
                path,
                expanded_nodes,
                path_cost(path, obstacle_points=obstacles, risk_radius=risk_radius, risk_weight=risk_weight),
            )

    return PathResult(RRT, None, expanded_nodes, None)


def bug2_path(
    grid_map: GridMap,
    start: Point,
    goal: Point,
    blocked_points: Iterable[Point] | None = None,
    obstacle_points: Iterable[Point] | None = None,
    risk_radius: int = 0,
    risk_weight: float = 0.0,
) -> PathResult:
    blocked = _sanitize_blocked_points(start, goal, blocked_points)
    obstacles = set(obstacle_points or [])
    m_line = set(_bresenham_line(start, goal))

    frontier: list[tuple[float, int, Point]] = []
    tie_breaker = count()
    heappush(frontier, (0.0, next(tie_breaker), start))
    came_from: dict[Point, Point | None] = {start: None}
    expanded_nodes = 0

    while frontier:
        _, _, current = heappop(frontier)
        expanded_nodes += 1
        if current == goal:
            path = _reconstruct_path(came_from, goal)
            return PathResult(
                BUG2,
                path,
                expanded_nodes,
                path_cost(path, obstacle_points=obstacles, risk_radius=risk_radius, risk_weight=risk_weight),
            )

        for neighbor in grid_map.neighbors(current):
            if neighbor in blocked or neighbor in came_from:
                continue
            came_from[neighbor] = current
            m_line_bias = 0 if neighbor in m_line else _distance_to_line(neighbor, start, goal)
            wall_bias = _wall_contact_count(grid_map, neighbor, blocked) * 0.15
            priority = manhattan_distance(neighbor, goal) + m_line_bias * 1.5 + wall_bias
            heappush(frontier, (priority, next(tie_breaker), neighbor))

    return PathResult(BUG2, None, expanded_nodes, None)


def potential_fields_path(
    grid_map: GridMap,
    start: Point,
    goal: Point,
    blocked_points: Iterable[Point] | None = None,
    obstacle_points: Iterable[Point] | None = None,
    risk_radius: int = 0,
    risk_weight: float = 0.0,
) -> PathResult:
    blocked = _sanitize_blocked_points(start, goal, blocked_points)
    obstacles = set(obstacle_points or [])
    current = start
    path = [start]
    visited: dict[Point, int] = {start: 1}
    expanded_nodes = 0
    max_steps = max(200, grid_map.rows * grid_map.cols * 6)

    for _ in range(max_steps):
        expanded_nodes += 1
        if current == goal:
            return PathResult(
                POTENTIAL_FIELDS,
                path,
                expanded_nodes,
                path_cost(path, obstacle_points=obstacles, risk_radius=risk_radius, risk_weight=risk_weight),
            )

        candidates = [neighbor for neighbor in grid_map.neighbors(current) if neighbor not in blocked]
        if not candidates:
            return PathResult(POTENTIAL_FIELDS, None, expanded_nodes, None)

        next_point = min(
            candidates,
            key=lambda point: (
                _potential_score(
                    point,
                    goal,
                    visited,
                    obstacle_points=obstacles,
                    risk_radius=risk_radius,
                    risk_weight=risk_weight,
                ),
                manhattan_distance(point, goal),
            ),
        )

        current = next_point
        path.append(current)
        visited[current] = visited.get(current, 0) + 1
        if visited[current] > 4:
            return PathResult(POTENTIAL_FIELDS, None, expanded_nodes, None)

    return PathResult(POTENTIAL_FIELDS, None, expanded_nodes, None)


def choose_next_target(
    robot_position: Point,
    stops: set[Point],
    goal: Point | None,
) -> Point | None:
    if goal is None:
        return None
    if stops:
        return min(stops, key=lambda point: manhattan_distance(robot_position, point))
    return goal


def compare_algorithms(
    grid_map: GridMap,
    start: Point,
    goal: Point,
    blocked_points: Iterable[Point] | None = None,
    obstacle_points: Iterable[Point] | None = None,
    risk_radius: int = 0,
    risk_weight: float = 0.0,
) -> dict[str, PathResult]:
    return {
        algorithm: find_path(
            algorithm,
            grid_map,
            start,
            goal,
            blocked_points,
            obstacle_points=obstacle_points,
            risk_radius=risk_radius,
            risk_weight=risk_weight,
        )
        for algorithm in (ASTAR, DIJKSTRA, GREEDY, RRT, BFS, WEIGHTED_ASTAR, BUG2, POTENTIAL_FIELDS)
    }


def movement_cost(
    point: Point,
    obstacle_points: Iterable[Point] | None = None,
    risk_radius: int = 0,
    risk_weight: float = 0.0,
) -> float:
    return 1.0 + risk_penalty(point, obstacle_points, risk_radius, risk_weight)


def risk_penalty(
    point: Point,
    obstacle_points: Iterable[Point] | None = None,
    risk_radius: int = 0,
    risk_weight: float = 0.0,
) -> float:
    obstacles = tuple(obstacle_points or ())
    if not obstacles or risk_radius <= 0 or risk_weight <= 0:
        return 0.0

    nearest_distance = min(manhattan_distance(point, obstacle) for obstacle in obstacles)
    if nearest_distance > risk_radius:
        return 0.0

    return risk_weight * (risk_radius - nearest_distance + 1)


def path_cost(
    path: list[Point] | None,
    obstacle_points: Iterable[Point] | None = None,
    risk_radius: int = 0,
    risk_weight: float = 0.0,
) -> float | None:
    if path is None:
        return None

    total = 0.0
    for point in path[1:]:
        total += movement_cost(
            point,
            obstacle_points=obstacle_points,
            risk_radius=risk_radius,
            risk_weight=risk_weight,
        )
    return total


def _sanitize_blocked_points(
    start: Point,
    goal: Point,
    blocked_points: Iterable[Point] | None,
) -> set[Point]:
    blocked = set(blocked_points or [])
    blocked.discard(start)
    blocked.discard(goal)
    return blocked


def _bresenham_line(start: Point, goal: Point) -> list[Point]:
    points: list[Point] = []
    row0, col0 = start.row, start.col
    row1, col1 = goal.row, goal.col
    delta_col = abs(col1 - col0)
    delta_row = -abs(row1 - row0)
    step_col = 1 if col0 < col1 else -1
    step_row = 1 if row0 < row1 else -1
    error = delta_col + delta_row

    while True:
        points.append(Point(row0, col0))
        if row0 == row1 and col0 == col1:
            return points
        doubled_error = 2 * error
        if doubled_error >= delta_row:
            error += delta_row
            col0 += step_col
        if doubled_error <= delta_col:
            error += delta_col
            row0 += step_row


def _best_towards_goal_step(
    grid_map: GridMap,
    current: Point,
    goal: Point,
    blocked: set[Point],
    visited: dict[Point, int],
) -> Point | None:
    current_distance = manhattan_distance(current, goal)
    candidates = [
        neighbor
        for neighbor in grid_map.neighbors(current)
        if neighbor not in blocked and manhattan_distance(neighbor, goal) < current_distance
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda point: (
            visited.get(point, 0),
            manhattan_distance(point, goal),
            point.row,
            point.col,
        ),
    )


def _wall_following_step(
    grid_map: GridMap,
    current: Point,
    goal: Point,
    blocked: set[Point],
    visited: dict[Point, int],
    m_line: set[Point],
) -> Point | None:
    candidates = [neighbor for neighbor in grid_map.neighbors(current) if neighbor not in blocked]
    if not candidates:
        return None
    current_distance = manhattan_distance(current, goal)
    return min(
        candidates,
        key=lambda point: (
            0 if point in m_line and manhattan_distance(point, goal) < current_distance else 1,
            visited.get(point, 0),
            _wall_contact_count(grid_map, point, blocked),
            manhattan_distance(point, goal),
            point.row,
            point.col,
        ),
    )


def _wall_contact_count(grid_map: GridMap, point: Point, blocked: set[Point]) -> int:
    contacts = 0
    for candidate in (
        Point(point.row - 1, point.col),
        Point(point.row + 1, point.col),
        Point(point.row, point.col - 1),
        Point(point.row, point.col + 1),
    ):
        if not grid_map.in_bounds(candidate) or not grid_map.is_free(candidate) or candidate in blocked:
            contacts += 1
    return contacts


def _distance_to_line(point: Point, start: Point, goal: Point) -> float:
    row_delta = goal.row - start.row
    col_delta = goal.col - start.col
    if row_delta == 0 and col_delta == 0:
        return 0.0

    numerator = abs(
        col_delta * (start.row - point.row)
        - (start.col - point.col) * row_delta
    )
    denominator = (row_delta * row_delta + col_delta * col_delta) ** 0.5
    return numerator / denominator


def _potential_score(
    point: Point,
    goal: Point,
    visited: dict[Point, int],
    obstacle_points: Iterable[Point] | None = None,
    risk_radius: int = 0,
    risk_weight: float = 0.0,
) -> float:
    attractive = manhattan_distance(point, goal)
    repulsive = risk_penalty(point, obstacle_points, risk_radius, risk_weight)
    revisit_penalty = visited.get(point, 0) * 3.0
    return attractive + repulsive + revisit_penalty


def _reconstruct_path(came_from: dict[Point, Point | None], goal: Point) -> list[Point]:
    path = [goal]
    current = goal
    while came_from[current] is not None:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _step_towards_sample(
    grid_map: GridMap,
    current: Point,
    sample: Point,
    blocked: set[Point],
) -> Point | None:
    candidates = [neighbor for neighbor in grid_map.neighbors(current) if neighbor not in blocked]
    if not candidates:
        return None
    return min(candidates, key=lambda point: manhattan_distance(point, sample))
