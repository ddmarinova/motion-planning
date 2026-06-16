from __future__ import annotations

from dataclasses import dataclass
import random
import tkinter as tk

from motion_planning.map import DEFAULT_TEMPLATE_ID, MAP_TEMPLATES, GridMap, Point
from motion_planning.planner import (
    ASTAR,
    BFS,
    BUG2,
    DIJKSTRA,
    GREEDY,
    PathResult,
    POTENTIAL_FIELDS,
    RRT,
    WEIGHTED_ASTAR,
    compare_algorithms,
    choose_next_target,
    find_path,
    risk_penalty,
)


CELL_SIZE = 24
ROWS = 25
COLS = 25

COLOR_BG = "#f4efe6"
COLOR_GRID = "#d7cfc1"
COLOR_FREE = "#fffaf0"
COLOR_WALL = "#34495e"
COLOR_START = "#f1c40f"
COLOR_GOAL = "#e74c3c"
COLOR_TEXT = "#2c3e50"
COLOR_OBSTACLE = "#8e44ad"
COLOR_GOAL_ICON = "#c0392b"
COLOR_PANEL = "#efe6d6"
COLOR_TABLE_HEADER = "#dde7e1"
COLOR_TABLE_ROW = "#fffaf0"
COLOR_TABLE_ALT_ROW = "#f3eadb"
COLOR_CHART_AXIS = "#b8ae9f"
COLOR_ASTAR_PATH = "#2980b9"
COLOR_DIJKSTRA_PATH = "#16a085"
COLOR_GREEDY_PATH = "#f1c40f"
COLOR_RRT_PATH = "#c0392b"
COLOR_BFS_PATH = "#7f8c8d"
COLOR_WEIGHTED_ASTAR_PATH = "#9b59b6"
COLOR_BUG2_PATH = "#e67e22"
COLOR_POTENTIAL_FIELDS_PATH = "#2ecc71"
COLOR_RISK = "#f6b26b"

ALGORITHMS = (ASTAR, DIJKSTRA, GREEDY, RRT, BFS, WEIGHTED_ASTAR, BUG2, POTENTIAL_FIELDS)
ALGORITHM_COLORS = {
    ASTAR: COLOR_ASTAR_PATH,
    DIJKSTRA: COLOR_DIJKSTRA_PATH,
    GREEDY: COLOR_GREEDY_PATH,
    RRT: COLOR_RRT_PATH,
    BFS: COLOR_BFS_PATH,
    WEIGHTED_ASTAR: COLOR_WEIGHTED_ASTAR_PATH,
    BUG2: COLOR_BUG2_PATH,
    POTENTIAL_FIELDS: COLOR_POTENTIAL_FIELDS_PATH,
}
ALGORITHM_LABELS = {
    ASTAR: "A*",
    DIJKSTRA: "Dijkstra",
    GREEDY: "Greedy",
    RRT: "RRT",
    BFS: "BFS",
    WEIGHTED_ASTAR: "Weighted A*",
    BUG2: "Bug 2",
    POTENTIAL_FIELDS: "Potential",
}
ALGORITHM_CHART_LABELS = {
    ASTAR: "A*",
    DIJKSTRA: "Dij",
    GREEDY: "Gre",
    RRT: "RRT",
    BFS: "BFS",
    WEIGHTED_ASTAR: "WA*",
    BUG2: "Bug",
    POTENTIAL_FIELDS: "Pot",
}
OBSTACLE_MOVE_INTERVAL_MS = 300
ROBOT_MOVE_INTERVAL_MS = 200
RISK_RADIUS = 3
RISK_WEIGHT = 2.5
OBSTACLE_EMOJIS = [
    "😀",
    "😃",
    "😄",
    "😁",
    "😆",
    "😊",
    "🙂",
    "😉",
    "😋",
    "😎",
    "🥳",
    "🤩",
    "😇",
    "🤠",
    "😺",
    "😸",
    "😹",
    "😻",
    "😼",
    "🙃",
]


@dataclass
class Obstacle:
    position: Point
    icon: str


class MotionPlanningApp:
    def __init__(self) -> None:
        self.grid_map = GridMap(rows=ROWS, cols=COLS)
        self.robot_position = self.grid_map.start
        self.obstacles: list[Obstacle] = []
        self.current_path: list[Point] = []
        self.active_algorithm = ASTAR
        self.comparison_results: dict[str, PathResult] = {}
        self.display_results: dict[str, PathResult] = {}
        self.goal_snapshot_results: dict[str, PathResult] = {}
        self.replan_counts = {algorithm: 0 for algorithm in ALGORITHMS}
        self.current_target: Point | None = None
        self.start_marker: Point | None = None
        self.simulation_complete = False
        self.rng = random.Random()
        self.width = COLS * CELL_SIZE
        self.height = ROWS * CELL_SIZE

        self.root = tk.Tk()
        self.root.title("Motion Planning Map")
        self.root.configure(bg=COLOR_BG)

        self.status_var = tk.StringVar(value="")
        self.algorithm_var = tk.StringVar(value=ASTAR)
        self.risk_enabled_var = tk.BooleanVar(value=True)
        self.template_name_by_id = {template.template_id: template.name for template in MAP_TEMPLATES}
        self.template_id_by_name = {template.name: template.template_id for template in MAP_TEMPLATES}
        self.map_template_var = tk.StringVar(value=self.template_name_by_id[DEFAULT_TEMPLATE_ID])

        self.main_frame = tk.Frame(self.root, bg=COLOR_BG)
        self.main_frame.pack(padx=16, pady=16)

        self.canvas = tk.Canvas(
            self.main_frame,
            width=self.width,
            height=self.height,
            bg=COLOR_BG,
            highlightthickness=0,
        )
        self.canvas.pack(side="left")

        self.sidebar_frame = tk.Frame(self.main_frame, bg=COLOR_BG, width=520)
        self.sidebar_frame.pack(side="left", padx=(16, 0), fill="y")
        self.sidebar_frame.pack_propagate(False)

        self.sidebar_canvas = tk.Canvas(
            self.sidebar_frame,
            width=500,
            height=self.height,
            bg=COLOR_BG,
            highlightthickness=0,
        )
        self.sidebar_scrollbar = tk.Scrollbar(
            self.sidebar_frame,
            orient="vertical",
            command=self.sidebar_canvas.yview,
        )
        self.sidebar_content = tk.Frame(self.sidebar_canvas, bg=COLOR_BG)
        self.sidebar_window = self.sidebar_canvas.create_window(
            (0, 0),
            window=self.sidebar_content,
            anchor="nw",
        )
        self.sidebar_canvas.configure(yscrollcommand=self.sidebar_scrollbar.set)
        self.sidebar_canvas.pack(side="left", fill="both", expand=True)
        self.sidebar_scrollbar.pack(side="right", fill="y")
        self.sidebar_content.bind("<Configure>", self._on_sidebar_configure)
        self.sidebar_canvas.bind("<Configure>", self._on_sidebar_canvas_configure)
        self.sidebar_canvas.bind_all("<MouseWheel>", self._on_sidebar_mousewheel)

        self.controls_frame = tk.Frame(self.sidebar_content, bg=COLOR_BG)
        self.controls_frame.pack(fill="x", pady=(0, 16))

        self.map_template_label = tk.Label(
            self.controls_frame,
            text="Карта:",
            bg=COLOR_BG,
            fg=COLOR_TEXT,
            font=("Helvetica", 16, "bold"),
            anchor="w",
        )
        self.map_template_label.pack(fill="x")

        self.map_template_menu = tk.OptionMenu(
            self.controls_frame,
            self.map_template_var,
            *self.template_id_by_name.keys(),
            command=self.set_map_template,
        )
        self.map_template_menu.configure(
            bg=COLOR_PANEL,
            fg=COLOR_TEXT,
            activebackground=COLOR_GRID,
            activeforeground=COLOR_TEXT,
            relief=tk.FLAT,
            font=("Helvetica", 16, "bold"),
            cursor="hand2",
            highlightthickness=0,
        )
        self.map_template_menu.pack(fill="x", pady=(6, 12))

        self.add_obstacle_button = tk.Button(
            self.controls_frame,
            text="Добави препятствие",
            command=self.add_obstacle,
            bg="#74b9ff",
            fg="#1f1f1f",
            activebackground="#4f97d7",
            activeforeground="#1f1f1f",
            relief=tk.FLAT,
            padx=14,
            pady=8,
            font=("Helvetica", 16, "bold"),
            cursor="hand2",
        )
        self.add_obstacle_button.pack(fill="x")

        self.algorithm_frame = tk.Frame(self.sidebar_content, bg=COLOR_BG)
        self.algorithm_frame.pack(fill="x")

        self.algorithm_label = tk.Label(
            self.algorithm_frame,
            text="Алгоритъм за движение:",
            bg=COLOR_BG,
            fg=COLOR_TEXT,
            font=("Helvetica", 16, "bold"),
        )
        self.algorithm_label.pack(anchor="w")

        self.astar_radio = tk.Radiobutton(
            self.algorithm_frame,
            text="A*",
            variable=self.algorithm_var,
            value=ASTAR,
            command=lambda: self.set_active_algorithm(ASTAR),
            bg=COLOR_BG,
            fg=COLOR_ASTAR_PATH,
            selectcolor=COLOR_PANEL,
            activebackground=COLOR_BG,
            activeforeground=COLOR_ASTAR_PATH,
            font=("Helvetica", 16),
        )
        self.astar_radio.pack(anchor="w", pady=(6, 0))

        self.dijkstra_radio = tk.Radiobutton(
            self.algorithm_frame,
            text="Dijkstra",
            variable=self.algorithm_var,
            value=DIJKSTRA,
            command=lambda: self.set_active_algorithm(DIJKSTRA),
            bg=COLOR_BG,
            fg=COLOR_DIJKSTRA_PATH,
            selectcolor=COLOR_PANEL,
            activebackground=COLOR_BG,
            activeforeground=COLOR_DIJKSTRA_PATH,
            font=("Helvetica", 16),
        )
        self.dijkstra_radio.pack(anchor="w")

        self.greedy_radio = tk.Radiobutton(
            self.algorithm_frame,
            text="Greedy Best-First",
            variable=self.algorithm_var,
            value=GREEDY,
            command=lambda: self.set_active_algorithm(GREEDY),
            bg=COLOR_BG,
            fg=COLOR_GREEDY_PATH,
            selectcolor=COLOR_PANEL,
            activebackground=COLOR_BG,
            activeforeground=COLOR_GREEDY_PATH,
            font=("Helvetica", 16),
        )
        self.greedy_radio.pack(anchor="w")

        self.rrt_radio = tk.Radiobutton(
            self.algorithm_frame,
            text="RRT",
            variable=self.algorithm_var,
            value=RRT,
            command=lambda: self.set_active_algorithm(RRT),
            bg=COLOR_BG,
            fg=COLOR_RRT_PATH,
            selectcolor=COLOR_PANEL,
            activebackground=COLOR_BG,
            activeforeground=COLOR_RRT_PATH,
            font=("Helvetica", 16),
        )
        self.rrt_radio.pack(anchor="w")

        self.bfs_radio = tk.Radiobutton(
            self.algorithm_frame,
            text="BFS",
            variable=self.algorithm_var,
            value=BFS,
            command=lambda: self.set_active_algorithm(BFS),
            bg=COLOR_BG,
            fg=COLOR_BFS_PATH,
            selectcolor=COLOR_PANEL,
            activebackground=COLOR_BG,
            activeforeground=COLOR_BFS_PATH,
            font=("Helvetica", 16),
        )
        self.bfs_radio.pack(anchor="w")

        self.weighted_astar_radio = tk.Radiobutton(
            self.algorithm_frame,
            text="Weighted A*",
            variable=self.algorithm_var,
            value=WEIGHTED_ASTAR,
            command=lambda: self.set_active_algorithm(WEIGHTED_ASTAR),
            bg=COLOR_BG,
            fg=COLOR_WEIGHTED_ASTAR_PATH,
            selectcolor=COLOR_PANEL,
            activebackground=COLOR_BG,
            activeforeground=COLOR_WEIGHTED_ASTAR_PATH,
            font=("Helvetica", 16),
        )
        self.weighted_astar_radio.pack(anchor="w")

        self.bug2_radio = tk.Radiobutton(
            self.algorithm_frame,
            text="Bug 2",
            variable=self.algorithm_var,
            value=BUG2,
            command=lambda: self.set_active_algorithm(BUG2),
            bg=COLOR_BG,
            fg=COLOR_BUG2_PATH,
            selectcolor=COLOR_PANEL,
            activebackground=COLOR_BG,
            activeforeground=COLOR_BUG2_PATH,
            font=("Helvetica", 16),
        )
        self.bug2_radio.pack(anchor="w")

        self.potential_fields_radio = tk.Radiobutton(
            self.algorithm_frame,
            text="Potential Fields",
            variable=self.algorithm_var,
            value=POTENTIAL_FIELDS,
            command=lambda: self.set_active_algorithm(POTENTIAL_FIELDS),
            bg=COLOR_BG,
            fg=COLOR_POTENTIAL_FIELDS_PATH,
            selectcolor=COLOR_PANEL,
            activebackground=COLOR_BG,
            activeforeground=COLOR_POTENTIAL_FIELDS_PATH,
            font=("Helvetica", 16),
        )
        self.potential_fields_radio.pack(anchor="w")

        self.risk_checkbox = tk.Checkbutton(
            self.algorithm_frame,
            text="Risk zones",
            variable=self.risk_enabled_var,
            command=self.on_toggle_risk_zones,
            bg=COLOR_BG,
            fg=COLOR_TEXT,
            selectcolor=COLOR_PANEL,
            activebackground=COLOR_BG,
            activeforeground=COLOR_TEXT,
            font=("Helvetica", 16),
        )
        self.risk_checkbox.pack(anchor="w", pady=(10, 0))

        self.comparison_title_label = tk.Label(
            self.sidebar_content,
            text="Сравнение",
            bg=COLOR_BG,
            fg=COLOR_TEXT,
            font=("Helvetica", 16, "bold"),
            anchor="w",
        )

        self.comparison_table = tk.Frame(
            self.sidebar_content,
            bg=COLOR_BG,
        )
        self.comparison_chart = tk.Canvas(
            self.sidebar_content,
            width=500,
            height=430,
            bg=COLOR_BG,
            highlightthickness=0,
        )

        self.canvas.bind("<Button-1>", self.on_left_click)
        self.root.bind("<KeyPress-c>", self.on_clear_goal)
        self.root.bind("<KeyPress-C>", self.on_clear_goal)
        self.root.bind("<Escape>", self.on_escape)

        self.draw()
        self.clear_algorithm_comparison()
        self.schedule_obstacle_step()
        self.schedule_robot_step()

    def run(self) -> None:
        self.root.mainloop()

    def _on_sidebar_configure(self, _event: tk.Event) -> None:
        self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all"))

    def _on_sidebar_canvas_configure(self, event: tk.Event) -> None:
        self.sidebar_canvas.itemconfigure(self.sidebar_window, width=event.width)

    def _on_sidebar_mousewheel(self, event: tk.Event) -> None:
        self.sidebar_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def draw(self) -> None:
        self.refresh_display_paths()
        self.canvas.delete("all")
        obstacle_positions = self._obstacle_positions()
        for row in range(self.grid_map.rows):
            for col in range(self.grid_map.cols):
                point = Point(row, col)
                color = COLOR_FREE if self.grid_map.is_free(point) else COLOR_WALL
                if color == COLOR_FREE:
                    color = self._cell_color_with_risk(point, obstacle_positions)
                x0 = col * CELL_SIZE
                y0 = row * CELL_SIZE
                x1 = x0 + CELL_SIZE
                y1 = y0 + CELL_SIZE
                self.canvas.create_rectangle(
                    x0,
                    y0,
                    x1,
                    y1,
                    fill=color,
                    outline=COLOR_GRID,
                    width=1,
                )

        self._draw_algorithm_paths()
        if self.start_marker is not None:
            self._draw_marker(self.start_marker, COLOR_START, "S")
        if self.grid_map.goal is not None:
            self._draw_marker(self.grid_map.goal, COLOR_GOAL_ICON, "G")
        self._draw_obstacles()
        self._draw_robot()

    def _cell_color_with_risk(self, point: Point, obstacle_positions: set[Point]) -> str:
        if not self.risk_enabled_var.get():
            return COLOR_FREE

        penalty = risk_penalty(
            point,
            obstacle_points=obstacle_positions,
            risk_radius=self._active_risk_radius(),
            risk_weight=self._active_risk_weight(),
        )
        if penalty <= 0:
            return COLOR_FREE

        normalized = min(1.0, penalty / (RISK_WEIGHT * (RISK_RADIUS + 1)))
        return self._blend_hex(COLOR_FREE, COLOR_RISK, normalized * 0.75)

    def _blend_hex(self, base: str, overlay: str, alpha: float) -> str:
        base_rgb = tuple(int(base[index:index + 2], 16) for index in (1, 3, 5))
        overlay_rgb = tuple(int(overlay[index:index + 2], 16) for index in (1, 3, 5))
        blended = tuple(
            round(base_channel * (1 - alpha) + overlay_channel * alpha)
            for base_channel, overlay_channel in zip(base_rgb, overlay_rgb)
        )
        return "#" + "".join(f"{channel:02x}" for channel in blended)

    def _draw_marker(self, point: Point, color: str, label: str) -> None:
        center_x = point.col * CELL_SIZE + CELL_SIZE / 2
        center_y = point.row * CELL_SIZE + CELL_SIZE / 2
        radius = 8
        self.canvas.create_oval(
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
            fill=color,
            outline="",
        )
        self.canvas.create_text(
            center_x,
            center_y,
            text=label,
            fill="#1f1f1f",
            font=("Helvetica", 15, "bold"),
        )

    def _draw_robot(self) -> None:
        point = self.robot_position
        center_x = point.col * CELL_SIZE + CELL_SIZE / 2
        center_y = point.row * CELL_SIZE + CELL_SIZE / 2
        self.canvas.create_text(
            center_x,
            center_y,
            text="🤖",
            font=("Apple Color Emoji", 22),
        )

    def _draw_algorithm_paths(self) -> None:
        for algorithm, color in ALGORITHM_COLORS.items():
            dash = () if algorithm == self.active_algorithm else (6, 4)
            self._draw_single_path(algorithm, color, dash=dash)

    def _draw_single_path(self, algorithm: str, color: str, dash: tuple[int, ...]) -> None:
        result = self.display_results.get(algorithm)
        if result is None or result.path is None or len(result.path) < 2:
            return

        coordinates: list[float] = []
        for point in result.path:
            center_x, center_y = self._cell_center(point)
            coordinates.extend((center_x, center_y))

        width = 4 if algorithm == self.active_algorithm else 2
        self.canvas.create_line(
            *coordinates,
            fill=color,
            width=width,
            dash=dash,
            capstyle=tk.ROUND,
            joinstyle=tk.ROUND,
        )

        for point in result.path[1:-1]:
            center_x, center_y = self._cell_center(point)
            radius = 2 if algorithm == self.active_algorithm else 1
            self.canvas.create_oval(
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
                fill=color,
                outline="",
            )

    def _draw_obstacles(self) -> None:
        for obstacle in self.obstacles:
            center_x = obstacle.position.col * CELL_SIZE + CELL_SIZE / 2
            center_y = obstacle.position.row * CELL_SIZE + CELL_SIZE / 2
            self.canvas.create_text(
                center_x,
                center_y,
                text=obstacle.icon,
                font=("Apple Color Emoji", 22),
            )

    def _cell_center(self, point: Point) -> tuple[float, float]:
        return (
            point.col * CELL_SIZE + CELL_SIZE / 2,
            point.row * CELL_SIZE + CELL_SIZE / 2,
        )

    def on_left_click(self, event: tk.Event) -> None:
        point = Point(event.y // CELL_SIZE, event.x // CELL_SIZE)
        if not self.grid_map.in_bounds(point):
            return

        if point in {obstacle.position for obstacle in self.obstacles}:
            return

        if self.grid_map.set_goal(point):
            self.simulation_complete = False
            self.start_marker = self.robot_position
            self.display_results = {}
            self.comparison_results = {}
            self.goal_snapshot_results = {}
            self.replan_counts = {algorithm: 0 for algorithm in ALGORITHMS}
            self.refresh_algorithm_comparison()
            self.goal_snapshot_results = dict(self.comparison_results)
        self.draw()

    def set_map_template(self, template_name: str) -> None:
        template_id = self.template_id_by_name.get(template_name)
        if template_id is None:
            return

        self.grid_map.apply_template(template_id)
        self.robot_position = self.grid_map.start
        self.obstacles.clear()
        self.current_path = []
        self.active_algorithm = ASTAR
        self.algorithm_var.set(ASTAR)
        self.start_marker = None
        self.replan_counts = {algorithm: 0 for algorithm in ALGORITHMS}
        self.simulation_complete = False

        self.clear_algorithm_comparison()
        self.draw()

    def on_clear_goal(self, _event: tk.Event) -> None:
        self.grid_map.clear_goal()
        self.current_path = []
        self.start_marker = None
        self.replan_counts = {algorithm: 0 for algorithm in ALGORITHMS}
        self.simulation_complete = False
        self.clear_algorithm_comparison()
        self.draw()

    def on_escape(self, _event: tk.Event) -> None:
        self.root.destroy()

    def add_obstacle(self) -> None:
        spawn_point = self.grid_map.random_free_point(self._blocked_points())
        if spawn_point is None:
            return

        self.obstacles.append(
            Obstacle(
                position=spawn_point,
                icon=self.rng.choice(OBSTACLE_EMOJIS),
            )
        )
        self.refresh_algorithm_comparison(count_replans=True)
        self.draw()

    def schedule_obstacle_step(self) -> None:
        self.move_obstacles()
        self.root.after(OBSTACLE_MOVE_INTERVAL_MS, self.schedule_obstacle_step)

    def move_obstacles(self) -> None:
        occupied_now = {obstacle.position for obstacle in self.obstacles}
        updated_positions: list[Point] = []
        moved = False

        for obstacle in self.obstacles:
            old_position = obstacle.position
            occupied_now.discard(obstacle.position)
            candidate_positions = [
                neighbor
                for neighbor in self.grid_map.neighbors(obstacle.position)
                if neighbor not in occupied_now
                and neighbor not in updated_positions
                and neighbor != self.robot_position
                and neighbor != self.grid_map.goal
            ]

            if candidate_positions:
                obstacle.position = self.rng.choice(candidate_positions)
                moved = moved or obstacle.position != old_position

            updated_positions.append(obstacle.position)
            occupied_now.add(obstacle.position)

        if self.obstacles:
            if moved and self.grid_map.goal is not None and not self.simulation_complete:
                self.refresh_algorithm_comparison(count_replans=True)
            self.draw()

    def schedule_robot_step(self) -> None:
        self.move_robot()
        self.root.after(ROBOT_MOVE_INTERVAL_MS, self.schedule_robot_step)

    def move_robot(self) -> None:
        if self.simulation_complete:
            return

        target = choose_next_target(
            robot_position=self.robot_position,
            stops=set(),
            goal=self.grid_map.goal,
        )
        if target is None:
            self.current_path = []
            return

        result = find_path(
            self.active_algorithm,
            self.grid_map,
            start=self.robot_position,
            goal=target,
            blocked_points=self._obstacle_positions(),
            obstacle_points=self._obstacle_positions(),
            risk_radius=self._active_risk_radius(),
            risk_weight=self._active_risk_weight(),
        )
        path = result.path
        if path is None or len(path) < 2:
            self.current_path = path or []
            self.draw()
            return

        self.current_path = path
        self.robot_position = path[1]

        self._handle_goal_if_reached()
        if self.simulation_complete:
            self.draw()
            return

        self.refresh_algorithm_comparison()
        self.draw()

    def _handle_goal_if_reached(self) -> None:
        if self.grid_map.goal is None:
            return
        if self.robot_position == self.grid_map.goal:
            self.simulation_complete = True
            self.current_path = []
            self.display_results = dict(self.goal_snapshot_results)

    def update_status_for_plan(self, prefix: str | None = None) -> None:
        return

    def set_active_algorithm(self, algorithm: str) -> None:
        self.active_algorithm = algorithm
        self.algorithm_var.set(algorithm)
        self.update_comparison_display()
        self.draw()

    def refresh_algorithm_comparison(self, count_replans: bool = False) -> None:
        results = self._calculate_algorithm_results(count_replans=count_replans)
        if not results:
            self.clear_algorithm_comparison()
            return

        self.current_target = self.grid_map.goal
        self.comparison_results = results
        self.update_comparison_display()

    def refresh_display_paths(self) -> None:
        if self.simulation_complete:
            return
        self.display_results = self._calculate_algorithm_results()

    def _calculate_algorithm_results(self, count_replans: bool = False) -> dict[str, PathResult]:
        if self.grid_map.goal is None:
            return {}

        target = choose_next_target(
            robot_position=self.robot_position,
            stops=set(),
            goal=self.grid_map.goal,
        )
        if target is None:
            return {}

        blocked_points = self._obstacle_positions()
        if count_replans:
            for algorithm in ALGORITHMS:
                self.replan_counts[algorithm] += 1

        return compare_algorithms(
            self.grid_map,
            start=self.robot_position,
            goal=target,
            blocked_points=blocked_points,
            obstacle_points=blocked_points,
            risk_radius=self._active_risk_radius(),
            risk_weight=self._active_risk_weight(),
        )

    def clear_algorithm_comparison(self) -> None:
        self.current_target = None
        self.comparison_results = {}
        self.display_results = {}
        self.goal_snapshot_results = {}
        self.replan_counts = {algorithm: 0 for algorithm in ALGORITHMS}
        self._hide_comparison_table()

    def on_toggle_risk_zones(self) -> None:
        if self.current_target is not None:
            self.refresh_algorithm_comparison()
        else:
            self.update_comparison_display()
        self.draw()

    def update_comparison_display(self) -> None:
        if self.current_target is None or not self.comparison_results:
            self._hide_comparison_table()
            return

        self._show_comparison_table()

    def _show_comparison_table(self) -> None:
        for child in self.comparison_table.winfo_children():
            child.destroy()

        if not self.comparison_title_label.winfo_ismapped():
            self.comparison_title_label.pack(fill="x", pady=(16, 0))
            self.comparison_table.pack(fill="x", pady=(8, 0))
            self.comparison_chart.pack(fill="x", pady=(12, 0))

        rows = tuple((ALGORITHM_LABELS[algorithm], algorithm) for algorithm in ALGORITHMS)
        table_results = self.goal_snapshot_results or self.comparison_results
        successful_lengths = [
            result.path_length
            for result in table_results.values()
            if result.path_length is not None
        ]
        best_length = min(successful_lengths) if successful_lengths else None

        headers = ("Алг.", "Статус", "Време", "Път", "Възли", "Цена", "Откл.", "Еф.", "Риск", "Преизч.")
        for column, header in enumerate(headers):
            self._comparison_cell(self.comparison_table, header, 0, column, bold=True, bg=COLOR_TABLE_HEADER)

        for row_index, (label, algorithm) in enumerate(rows, start=1):
            result = table_results[algorithm]
            status = "OK" if result.path is not None else "Fail"
            path_length = "-" if result.path_length is None else str(result.path_length)
            total_cost = "-" if result.total_cost is None else f"{result.total_cost:.1f}"
            deviation = self._format_path_deviation(result.path_length, best_length)
            efficiency = self._format_efficiency(result.path_length, result.expanded_nodes)
            risk = self._format_path_risk(result)
            row_bg = COLOR_TABLE_ROW if row_index % 2 else COLOR_TABLE_ALT_ROW
            self._comparison_cell(self.comparison_table, label, row_index, 0, fg=ALGORITHM_COLORS[algorithm], bold=True, bg=row_bg)
            self._comparison_cell(self.comparison_table, status, row_index, 1, fg="#1f7a3a" if result.path is not None else "#c0392b", bold=True, bg=row_bg)
            self._comparison_cell(self.comparison_table, f"{result.elapsed_ms:.1f}", row_index, 2, bg=row_bg)
            self._comparison_cell(self.comparison_table, path_length, row_index, 3, bg=row_bg)
            self._comparison_cell(self.comparison_table, str(result.expanded_nodes), row_index, 4, bg=row_bg)
            self._comparison_cell(self.comparison_table, total_cost, row_index, 5, bg=row_bg)
            self._comparison_cell(self.comparison_table, deviation, row_index, 6, bg=row_bg)
            self._comparison_cell(self.comparison_table, efficiency, row_index, 7, bg=row_bg)
            self._comparison_cell(self.comparison_table, risk, row_index, 8, bg=row_bg)
            self._comparison_cell(self.comparison_table, str(self.replan_counts[algorithm]), row_index, 9, bg=row_bg)

        self._draw_comparison_charts(table_results)

    def _format_path_deviation(self, path_length: int | None, best_length: int | None) -> str:
        if path_length is None or best_length is None:
            return "-"
        return f"+{path_length - best_length}"

    def _format_efficiency(self, path_length: int | None, expanded_nodes: int) -> str:
        if path_length is None or expanded_nodes <= 0:
            return "-"
        return f"{path_length / expanded_nodes:.2f}"

    def _format_path_risk(self, result: PathResult) -> str:
        if result.path_length is None or result.total_cost is None:
            return "-"
        risk = result.total_cost - result.path_length
        return f"{risk:.1f}"

    def _comparison_cell(
        self,
        parent: tk.Frame,
        text: str,
        row: int,
        column: int,
        fg: str = COLOR_TEXT,
        bold: bool = False,
        bg: str = COLOR_BG,
    ) -> None:
        label = tk.Label(
            parent,
            text=text,
            bg=bg,
            fg=fg,
            font=("Menlo", 11, "bold" if bold else "normal"),
            padx=5,
            pady=6,
            anchor="w",
            borderwidth=1,
            relief=tk.SOLID,
        )
        label.grid(row=row, column=column, sticky="nsew")
        parent.grid_columnconfigure(column, weight=1)

    def _draw_comparison_charts(self, results: dict[str, PathResult]) -> None:
        self.comparison_chart.delete("all")
        chart_specs = (
            ("Време", [results[algorithm].elapsed_ms for algorithm in ALGORITHMS], "ms"),
            (
                "Път",
                [results[algorithm].path_length or 0 for algorithm in ALGORITHMS],
                "",
            ),
            ("Възли", [results[algorithm].expanded_nodes for algorithm in ALGORITHMS], ""),
        )

        y = 8
        for title, values, suffix in chart_specs:
            self._draw_bar_chart(title, values, suffix, y)
            y += 138

    def _draw_bar_chart(self, title: str, values: list[float | int], suffix: str, y: int) -> None:
        left = 72
        top = y + 18
        bar_height = 8
        max_width = 350
        max_value = max(values) if values else 0
        scale = max_width / max_value if max_value else 0

        self.comparison_chart.create_text(
            0,
            y,
            text=title,
            fill=COLOR_TEXT,
            font=("Helvetica", 13, "bold"),
            anchor="nw",
        )
        self.comparison_chart.create_line(left, top - 4, left + max_width, top - 4, fill=COLOR_CHART_AXIS)

        for index, (algorithm, value) in enumerate(zip(ALGORITHMS, values)):
            label = ALGORITHM_CHART_LABELS[algorithm]
            bar_y = top + index * 12
            width = max(1, value * scale) if value else 0
            self.comparison_chart.create_text(
                0,
                bar_y + bar_height / 2,
                text=label,
                fill=ALGORITHM_COLORS[algorithm],
                font=("Helvetica", 12, "bold"),
                anchor="w",
            )
            self.comparison_chart.create_rectangle(
                left,
                bar_y,
                left + width,
                bar_y + bar_height,
                fill=ALGORITHM_COLORS[algorithm],
                outline="",
            )
            display_value = f"{value:.1f}{suffix}" if isinstance(value, float) else f"{value}{suffix}"
            self.comparison_chart.create_text(
                left + max_width + 4,
                bar_y + bar_height / 2,
                text=display_value,
                fill=COLOR_TEXT,
                font=("Helvetica", 12),
                anchor="w",
            )

    def _hide_comparison_table(self) -> None:
        for child in self.comparison_table.winfo_children():
            child.destroy()
        self.comparison_chart.delete("all")
        self.comparison_title_label.pack_forget()
        self.comparison_table.pack_forget()
        self.comparison_chart.pack_forget()

    def _algorithm_display_name(self, algorithm: str) -> str:
        return ALGORITHM_LABELS.get(algorithm, algorithm)

    def _blocked_points(self) -> set[Point]:
        blocked = {self.robot_position}
        if self.grid_map.goal is not None:
            blocked.add(self.grid_map.goal)
        blocked.update(self._obstacle_positions())
        return blocked

    def _obstacle_positions(self) -> set[Point]:
        return {obstacle.position for obstacle in self.obstacles}

    def _active_risk_radius(self) -> int:
        return RISK_RADIUS if self.risk_enabled_var.get() else 0

    def _active_risk_weight(self) -> float:
        return RISK_WEIGHT if self.risk_enabled_var.get() else 0.0


def run() -> None:
    app = MotionPlanningApp()
    app.run()
