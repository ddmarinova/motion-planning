from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Iterable


WALL = 1
FREE = 0


@dataclass(frozen=True)
class Point:
    row: int
    col: int


@dataclass(frozen=True)
class MapTemplate:
    template_id: str
    name: str
    description: str
    builder: Callable[[int, int], list[list[int]]]


def _empty_cells(rows: int, cols: int) -> list[list[int]]:
    cells = [[FREE for _ in range(cols)] for _ in range(rows)]
    _add_walls(cells, rows, cols)
    return cells


def _add_walls(cells: list[list[int]], rows: int, cols: int) -> None:
    for row in range(rows):
        cells[row][0] = WALL
        cells[row][cols - 1] = WALL
    for col in range(cols):
        cells[0][col] = WALL
        cells[rows - 1][col] = WALL


def _add_vertical_wall(cells: list[list[int]], col: int, start: int, end: int, gaps: set[int]) -> None:
    for row in range(start, end + 1):
        if row not in gaps:
            cells[row][col] = WALL


def _add_horizontal_wall(cells: list[list[int]], row: int, start: int, end: int, gaps: set[int]) -> None:
    for col in range(start, end + 1):
        if col not in gaps:
            cells[row][col] = WALL


def _warehouse_template(rows: int, cols: int) -> list[list[int]]:
    cells = _empty_cells(rows, cols)
    shelf_columns = (3, 4, 8, 9, 15, 16, 20, 21)
    for col in shelf_columns:
        for row in range(2, rows - 2):
            if row not in {6, 12, 18}:
                cells[row][col] = WALL
    return cells


def _office_template(rows: int, cols: int) -> list[list[int]]:
    cells = _empty_cells(rows, cols)

    _add_vertical_wall(cells, 8, 2, rows - 3, {5, 12, 19})
    _add_vertical_wall(cells, 16, 2, rows - 3, {5, 12, 19})
    _add_horizontal_wall(cells, 8, 2, cols - 3, {5, 12, 19})
    _add_horizontal_wall(cells, 16, 2, cols - 3, {5, 12, 19})

    for point in (
        Point(1, 8),
        Point(1, 16),
        Point(8, 1),
        Point(8, 23),
        Point(16, 1),
        Point(16, 23),
        Point(23, 8),
        Point(23, 16),
    ):
        cells[point.row][point.col] = WALL

    return cells


def _maze_template(rows: int, cols: int) -> list[list[int]]:
    cells = [[WALL for _ in range(cols)] for _ in range(rows)]
    rng = random.Random(42)
    start = Point(1, 1)
    stack = [start]
    cells[start.row][start.col] = FREE

    while stack:
        current = stack[-1]
        neighbors: list[tuple[Point, Point]] = []
        for row_delta, col_delta in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            next_point = Point(current.row + row_delta, current.col + col_delta)
            wall_between = Point(current.row + row_delta // 2, current.col + col_delta // 2)
            if (
                1 <= next_point.row < rows - 1
                and 1 <= next_point.col < cols - 1
                and cells[next_point.row][next_point.col] == WALL
            ):
                neighbors.append((next_point, wall_between))

        if not neighbors:
            stack.pop()
            continue

        next_point, wall_between = rng.choice(neighbors)
        cells[wall_between.row][wall_between.col] = FREE
        cells[next_point.row][next_point.col] = FREE
        stack.append(next_point)

    openings = (
        Point(1, cols - 2),
        Point(rows - 2, 1),
        Point(rows - 2, cols - 2),
    )
    for opening in openings:
        cells[opening.row][opening.col] = FREE

    return cells


def _factory_template(rows: int, cols: int) -> list[list[int]]:
    cells = _empty_cells(rows, cols)
    _add_horizontal_wall(cells, 6, 2, cols - 3, {4, 12, 20})
    _add_horizontal_wall(cells, 18, 2, cols - 3, {4, 12, 20})
    for row in (10, 11, 13, 14):
        for col in range(4, cols - 6):
            mirror_col = cols - 1 - col
            if col not in {11, 12, 13}:
                cells[row][col] = WALL
                cells[row][mirror_col] = WALL
    return cells


MAP_TEMPLATES = [
    MapTemplate("warehouse", "Склад", "Рафтове с широки коридори.", _warehouse_template),
    MapTemplate("office", "Офис", "Стаи, коридори и врати.", _office_template),
    MapTemplate("maze", "Лабиринт", "По-тесни проходи за по-трудно планиране.", _maze_template),
    MapTemplate("factory", "Производство", "Зони с машини и централни проходи.", _factory_template),
]
DEFAULT_TEMPLATE_ID = MAP_TEMPLATES[0].template_id


class GridMap:
    def __init__(
        self,
        rows: int,
        cols: int,
        wall_probability: float = 0.22,
        seed: int | None = None,
        template_id: str = DEFAULT_TEMPLATE_ID,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.wall_probability = wall_probability
        self._rng = random.Random(seed)
        self.start = Point(rows // 2, cols // 2)
        self.goal: Point | None = None
        self.stops: set[Point] = set()
        self.cells: list[list[int]] = []
        self.template_id = template_id
        self.regenerate()

    def regenerate(self) -> None:
        template = self._template_by_id(self.template_id)
        if template is not None:
            self.cells = template.builder(self.rows, self.cols)
            self._ensure_free(self.start)
            self._clear_start_region(radius=1)
            self.goal = None
            self.stops.clear()
            return

        self.cells = [
            [
                WALL if self._rng.random() < self.wall_probability else FREE
                for _ in range(self.cols)
            ]
            for _ in range(self.rows)
        ]

        self._carve_border()
        self._ensure_free(self.start)
        self._clear_start_region(radius=1)
        self.goal = None
        self.stops.clear()

    def apply_template(self, template_id: str) -> None:
        if self._template_by_id(template_id) is None:
            return
        self.template_id = template_id
        self.regenerate()

    def _template_by_id(self, template_id: str) -> MapTemplate | None:
        for template in MAP_TEMPLATES:
            if template.template_id == template_id:
                return template
        return None

    def _carve_border(self) -> None:
        for row in range(self.rows):
            self.cells[row][0] = FREE
            self.cells[row][self.cols - 1] = FREE
        for col in range(self.cols):
            self.cells[0][col] = FREE
            self.cells[self.rows - 1][col] = FREE

    def _clear_start_region(self, radius: int) -> None:
        for row in range(max(0, self.start.row - radius), min(self.rows, self.start.row + radius + 1)):
            for col in range(max(0, self.start.col - radius), min(self.cols, self.start.col + radius + 1)):
                self.cells[row][col] = FREE

    def in_bounds(self, point: Point) -> bool:
        return 0 <= point.row < self.rows and 0 <= point.col < self.cols

    def is_free(self, point: Point) -> bool:
        return self.in_bounds(point) and self.cells[point.row][point.col] == FREE

    def neighbors(self, point: Point) -> list[Point]:
        candidates = (
            Point(point.row - 1, point.col),
            Point(point.row + 1, point.col),
            Point(point.row, point.col - 1),
            Point(point.row, point.col + 1),
        )
        return [candidate for candidate in candidates if self.is_free(candidate)]

    def set_goal(self, point: Point) -> bool:
        if not self.is_free(point):
            return False
        self.goal = point
        return True

    def clear_goal(self) -> None:
        self.goal = None

    def has_stop(self, point: Point) -> bool:
        return point in self.stops

    def add_stop(self, point: Point) -> bool:
        if not self.is_free(point) or point == self.start or point == self.goal:
            return False
        self.stops.add(point)
        return True

    def remove_stop(self, point: Point) -> bool:
        if point in self.stops:
            self.stops.remove(point)
            return True
        return False

    def toggle_stop(self, point: Point) -> bool:
        if self.has_stop(point):
            self.stops.remove(point)
            return False
        self.add_stop(point)
        return True

    def clear_stops(self) -> None:
        self.stops.clear()

    def random_free_point(self, blocked_points: Iterable[Point] | None = None) -> Point | None:
        blocked = set(blocked_points or [])
        available = [
            Point(row, col)
            for row in range(self.rows)
            for col in range(self.cols)
            if self.cells[row][col] == FREE and Point(row, col) not in blocked
        ]
        if not available:
            return None
        return self._rng.choice(available)

    def _ensure_free(self, point: Point) -> None:
        self.cells[point.row][point.col] = FREE
