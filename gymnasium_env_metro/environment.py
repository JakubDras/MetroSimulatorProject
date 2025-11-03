import pygame
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import networkx as nx
from typing import List

from . import config
from .data_models import StationModel, LineModel, TrainModel, PassengerModel
from .entities import Station, Line, Train


class MiniMetroEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    #Metody Pomocnicze
    def _initialize_pygame(self):
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        pygame.display.set_caption("Mini Metro")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)
        self.font_large = pygame.font.Font(None, 72)

    def _get_info(self):
        return {"score": self.score, "week_number": self.week_number, "available_trains": self.available_trains}

    def _get_new_station_id(self) -> str:
        self.station_id_counter += 1
        return f"S{self.station_id_counter}"

    def _get_new_train_id(self) -> str:
        self.train_id_counter += 1
        return f"T{self.train_id_counter}"

    def _generate_initial_stations(self):
        initial_stations_count = 3

        k = min(initial_stations_count, len(config.STATION_SHAPES))
        initial_shapes = random.sample(config.STATION_SHAPES, k=k)

        first_x = random.randint(int(config.SCREEN_WIDTH * 0.25), int(config.SCREEN_WIDTH * 0.75))
        first_y = random.randint(int(config.SCREEN_HEIGHT * 0.25), int(config.SCREEN_HEIGHT * 0.75))

        station_model = StationModel(
            station_id=self._get_new_station_id(),
            pos=(first_x, first_y),
            shape=initial_shapes.pop()
        )
        new_station = Station(station_model)
        self.stations.append(new_station)
        self.G.add_node(station_model.station_id, pos=station_model.pos)
        #print(f"Dodano początkową stację: {new_station.data.shape} w ({first_x}, {first_y})")

        while len(self.stations) < initial_stations_count:
            neighbour_station = random.choice(self.stations)

            spawned = False
            attempts = 0
            while not spawned and attempts < 100:
                radius = config.SCREEN_WIDTH * 0.15

                x = random.randint(int(neighbour_station.pos.x - radius), int(neighbour_station.pos.x + radius))
                y = random.randint(int(neighbour_station.pos.y - radius), int(neighbour_station.pos.y + radius))

                if not (50 < x < config.SCREEN_WIDTH - 50 and 50 < y < config.SCREEN_HEIGHT - 50):
                    attempts += 1
                    continue

                new_pos_vec = pygame.Vector2(x, y)
                if all(new_pos_vec.distance_to(s.pos) > config.STATION_MIN_DISTANCE for s in self.stations):
                    station_model = StationModel(
                        station_id=self._get_new_station_id(),
                        pos=(x, y),
                        shape=initial_shapes.pop()
                    )
                    new_station = Station(station_model)
                    self.stations.append(new_station)
                    self.G.add_node(station_model.station_id, pos=station_model.pos)
                    #print(f"Dodano początkową stację: {new_station.data.shape} w ({x}, {y})")
                    spawned = True

                attempts += 1

    def _handle_manage_line(self, params) -> float:
        line_action_type, p1, p2 = params
        selected_color = config.LINE_COLORS[self.selected_line_index]
        current_line = next((line for line in self.lines if line.data.color == selected_color), None)

        if line_action_type == 0:
            if not (0 <= p1 < len(self.stations) and 0 <= p2 < len(self.stations) and p1 != p2):
                return -0.1

            s1, s2 = self.stations[p1], self.stations[p2]
            distance = s1.pos.distance_to(s2.pos)

            if current_line is None:
                if self.available_trains <= 0: return -0.2

                line_model = LineModel(color=selected_color, station_ids=[s1.data.station_id, s2.data.station_id])
                new_line = Line(line_model)
                new_line.stations.extend([s1, s2])
                self.lines.append(new_line)

                self.G.add_edge(s1.data.station_id, s2.data.station_id, key=str(selected_color), color=selected_color,
                                weight=distance)

                train_model = TrainModel(
                    train_id=self._get_new_train_id(), line_color=selected_color,
                    current_station_id=s1.data.station_id, target_station_id=s2.data.station_id,
                    pos=s1.data.pos
                )
                self.trains.append(Train(train_model, new_line))
                self.available_trains -= 1

                self.all_passengers_plan_update()  # <-- DODANO TUTAJ
                return 0.5
            else:
                success = False
                if current_line.stations[-1].data.station_id == s1.data.station_id and current_line.add_station(s2):
                    success = True
                elif current_line.stations[0].data.station_id == s2.data.station_id and current_line.add_station(s1,
                                                                                                                 index=0):
                    success = True

                if success:
                    current_line.data.station_ids = [s.data.station_id for s in current_line.stations]
                    self.G.add_edge(s1.data.station_id, s2.data.station_id, key=str(selected_color),
                                    color=selected_color, weight=distance)

                    self.all_passengers_plan_update()  # <-- DODANO TUTAJ
                    return 0.1

        elif line_action_type == 1:  # Usuń linię
            if current_line:
                station_objects_on_line = current_line.stations
                color_key = str(current_line.data.color)

                for i in range(len(station_objects_on_line) - 1):
                    u_id = station_objects_on_line[i].data.station_id
                    v_id = station_objects_on_line[i + 1].data.station_id

                    if self.G.has_edge(u_id, v_id, key=color_key):
                        self.G.remove_edge(u_id, v_id, key=color_key)

                trains_to_remove = [t for t in self.trains if t.line == current_line]
                for train in trains_to_remove:
                    last_station = next(
                        (s for s in self.stations if s.data.station_id == train.data.current_station_id),
                        None
                    )
                    if last_station and train.data.passengers:
                        last_station.data.passengers.extend(train.data.passengers)

                self.trains = [t for t in self.trains if t.line != current_line]
                self.available_trains += len(trains_to_remove)

                self.lines.remove(current_line)

                self.all_passengers_plan_update()  # <-- DODANO TUTAJ
                return 0.1

        return -0.05

    def _handle_deploy_train(self, params) -> float:
        station_idx = params[1]
        if self.available_trains > 0 and 0 <= station_idx < len(self.stations):
            target_station_obj = self.stations[station_idx]
            for line in self.lines:
                station_ids_on_line = [s.data.station_id for s in line.stations]
                if target_station_obj.data.station_id in station_ids_on_line and len(line.stations) >= 2:
                    current_idx = station_ids_on_line.index(target_station_obj.data.station_id)
                    next_id = None
                    if line.data.is_loop:
                        next_id = line.stations[(current_idx + 1) % (len(line.stations) - 1)].data.station_id
                    elif current_idx + 1 < len(line.stations):
                        next_id = line.stations[current_idx + 1].data.station_id
                    elif current_idx - 1 >= 0:
                        next_id = line.stations[current_idx - 1].data.station_id

                    if next_id:
                        train_model = TrainModel(
                            train_id=self._get_new_train_id(),
                            line_color=line.data.color,
                            current_station_id=target_station_obj.data.station_id,
                            target_station_id=next_id,
                            pos=target_station_obj.data.pos
                        )
                        self.trains.append(Train(train_model, line))
                        self.available_trains -= 1
                        return 0.2
        return -0.1

    def _handle_select_line(self, params) -> float:
        color_idx = params[1]
        if 0 <= color_idx < len(config.LINE_COLORS):
            self.selected_line_index = color_idx
            return 0.01
        return -0.01

    def _update_passenger_spawning(self):
        self.passenger_spawn_timer += 1

        if self.passenger_spawn_timer >= config.PASSENGER_SPAWN_RATE_FRAMES:
            self.passenger_spawn_timer = 0
            if not self.stations: return
            station = random.choice(self.stations)

            if len(station.data.passengers) < station.capacity:
                possible_targets = [s.data.shape for s in self.stations if s.data.shape != station.data.shape]
                if not possible_targets: return

                target_shape = random.choice(possible_targets)
                p_model = PassengerModel(origin_station_id=station.data.station_id, target_shape=target_shape)
                self.travel_planner_for_new_passenger(p_model)
                station.data.passengers.append(p_model)
            else:
                station.data.is_overcrowded = True
                self.game_over = True

    def _update_trains(self) -> float:
        reward = 0
        trains_to_remove = [t for t in self.trains if not t.update(self)]
        if trains_to_remove:
            self.trains = [t for t in self.trains if t not in trains_to_remove]
            reward -= 0.5 * len(trains_to_remove)
        return reward

    def _update_station_spawning(self):
        if len(self.stations) >= config.MAX_STATIONS or self.spawned_stations_this_week >= config.STATIONS_TO_SPAWN_PER_WEEK:
            return
        if self.week_timer < config.STATION_SPAWN_TIMES[self.spawned_stations_this_week]:
            return

        spawned = False
        attempts = 0
        max_attempts = 100

        while not spawned and attempts < max_attempts:
            if not self.stations: return

            neighbour_station = random.choice(self.stations)

            x_radius = config.SCREEN_WIDTH * 0.12
            y_radius = config.SCREEN_HEIGHT * 0.12

            x = random.randint(round(neighbour_station.pos.x - x_radius), round(neighbour_station.pos.x + x_radius))
            y = random.randint(round(neighbour_station.pos.y - y_radius), round(neighbour_station.pos.y + y_radius))

            while not (20 < x < config.SCREEN_WIDTH - 20 and 20 < y < config.SCREEN_HEIGHT - 20):
                x = random.randint(round(neighbour_station.pos.x - x_radius), round(neighbour_station.pos.x + x_radius))
                y = random.randint(round(neighbour_station.pos.y - y_radius), round(neighbour_station.pos.y + y_radius))

            new_pos_vec = pygame.Vector2(x, y)

            if all(new_pos_vec.distance_to(s.pos) > config.STATION_MIN_DISTANCE for s in self.stations):
                shape = random.choice(config.STATION_SHAPES)

                station_model = StationModel(
                    station_id=self._get_new_station_id(),
                    pos=(x, y),
                    shape=shape
                )
                new_station = Station(station_model)

                self.stations.append(new_station)
                self.G.add_node(station_model.station_id, pos=station_model.pos)

                self.all_passengers_plan_update()

                #print(f"Nowa stacja ({shape}) pojawiła się w Tygodniu {self.week_number}.")
                spawned = True
                self.spawned_stations_this_week += 1

            attempts += 1

        if not spawned and len(self.stations) < config.MAX_STATIONS:
            #print(f"Nie udało się dodać nowej stacji w Tygodniu {self.week_number} po {max_attempts} próbach.")
            pass

    def _update_week_timer(self) -> float:
        self.week_timer += 1
        if self.week_timer >= config.WEEK_DURATION_FRAMES:
            self.week_timer = 0
            self.week_number += 1
            self.available_trains += 1
            self.spawned_stations_this_week = 0
            return 10.0
        return 0.0

    def _draw_ui(self):
        self.ui_circles_rects = []
        start_x, y, radius, spacing = 20, 30, 15, 50
        for i, color in enumerate(config.LINE_COLORS):
            line_exists = any(line.data.color == color for line in self.lines)
            rect = pygame.Rect(start_x + i * spacing - radius, y - radius, radius * 2, radius * 2)
            self.ui_circles_rects.append(rect)
            pygame.draw.circle(self.screen, color, rect.center, radius)
            if i == self.selected_line_index:
                pygame.draw.circle(self.screen, config.WHITE, rect.center, radius + 2, 3)
            elif line_exists:
                pygame.draw.circle(self.screen, config.WHITE, rect.center, radius, 1)
            else:
                pygame.draw.circle(self.screen, config.GRAY, rect.center, radius, 1)

        score_text = self.font.render(f"Score: {self.score}", True, config.WHITE)
        self.screen.blit(score_text, (config.SCREEN_WIDTH - 180, 10))
        week_text = self.font.render(f"Week: {self.week_number}", True, config.WHITE)
        self.screen.blit(week_text, (config.SCREEN_WIDTH - 180, 40))

        pygame.draw.rect(self.screen, config.WEEK_BAR_BG_COLOR, config.WEEK_BAR_RECT)
        progress = self.week_timer / config.WEEK_DURATION_FRAMES
        current_bar_width = int(config.WEEK_BAR_RECT.width * progress)
        current_bar_rect = pygame.Rect(config.WEEK_BAR_RECT.left, config.WEEK_BAR_RECT.top, current_bar_width,
                                       config.WEEK_BAR_RECT.height)
        pygame.draw.rect(self.screen, config.WEEK_BAR_COLOR, current_bar_rect)
        pygame.draw.rect(self.screen, config.WHITE, config.WEEK_BAR_RECT, 1)

        button_color = config.GREEN if hasattr(self, 'deploy_train_mode') and self.deploy_train_mode else config.BLUE
        pygame.draw.rect(self.screen, button_color, config.TRAIN_BUTTON_RECT)
        pygame.draw.rect(self.screen, config.WHITE, config.TRAIN_BUTTON_RECT, 3)
        train_button_text = self.font_small.render(f"Train ({self.available_trains})", True, config.WHITE)
        text_rect = train_button_text.get_rect(center=config.TRAIN_BUTTON_RECT.center)
        self.screen.blit(train_button_text, text_rect)

    def _draw_game_over(self):
        overlay = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        game_over_text = self.font_large.render("Game Over", True, config.RED)
        score_text = self.font.render(f"Final Score: {self.score}", True, config.WHITE)
        week_text = self.font.render(f"Reached Week: {self.week_number}", True, config.WHITE)
        game_over_rect = game_over_text.get_rect(center=(config.SCREEN_WIDTH / 2, config.SCREEN_HEIGHT / 2 - 40))
        score_rect = score_text.get_rect(center=(config.SCREEN_WIDTH / 2, config.SCREEN_HEIGHT / 2 + 20))
        week_rect = week_text.get_rect(center=(config.SCREEN_WIDTH / 2, config.SCREEN_HEIGHT / 2 + 50))
        self.screen.blit(game_over_text, game_over_rect)
        self.screen.blit(score_text, score_rect)
        self.screen.blit(week_text, week_rect)

    def travel_planner_for_new_passenger(self, passenger_model: PassengerModel):
        best_path = None
        min_distance = float('inf')

        try:
            distances, paths = nx.single_source_dijkstra(
                self.G,
                source=passenger_model.origin_station_id,
                weight='weight'
            )
        except (nx.NetworkXNoPath, KeyError):
            passenger_model.travel_list = []
            return

        target_stations = [s for s in self.stations if s.data.shape == passenger_model.target_shape]

        for station in target_stations:
            station_id = station.data.station_id

            if station_id in distances:
                distance = distances[station_id]

                if distance < min_distance:
                    min_distance = distance
                    best_path = paths[station_id]

        if best_path:
            passenger_model.travel_list = best_path[1:]
        else:
            passenger_model.travel_list = []

    def all_passengers_plan_update(self):
        all_paths_from_all_sources = {}

        for station_obj in self.stations:
            source_id = station_obj.data.station_id

            if source_id not in all_paths_from_all_sources:
                try:
                    distances, paths = nx.single_source_dijkstra(
                        self.G,
                        source=source_id,
                        weight='weight'
                    )
                    all_paths_from_all_sources[source_id] = (distances, paths)
                except (nx.NetworkXNoPath, KeyError):
                    all_paths_from_all_sources[source_id] = (None, None)

            distances, paths = all_paths_from_all_sources[source_id]

            if distances is None:
                for p_model in station_obj.data.passengers:
                    p_model.travel_list = []
                continue

            best_paths_for_shapes = {}
            min_distances_for_shapes = {}

            for target_station in self.stations:
                target_id = target_station.data.station_id
                target_shape = target_station.data.shape

                if target_id == source_id:
                    continue

                if target_id in distances:
                    distance = distances[target_id]

                    current_min_dist = min_distances_for_shapes.get(target_shape, float('inf'))

                    if distance < current_min_dist:
                        min_distances_for_shapes[target_shape] = distance
                        best_paths_for_shapes[target_shape] = paths[target_id]
                    elif distance == current_min_dist:
                        if random.choice([True, False]):
                            best_paths_for_shapes[target_shape] = paths[target_id]

            for p_model in station_obj.data.passengers:
                best_path = best_paths_for_shapes.get(p_model.target_shape)

                if best_path:
                    p_model.travel_list = best_path[1:]
                else:
                    p_model.travel_list = []

    def _get_action_masks(self) -> dict:
        """Generuje i zwraca słownik masek dla wszystkich poziomów akcji."""
        num_current_stations = len(self.stations)

        # --- Inicjalizacja masek ---
        manage_line_mask = np.zeros((3, config.MAX_STATIONS, config.MAX_STATIONS), dtype=np.int8)
        manage_line_type_mask = np.zeros(3, dtype=np.int8)
        deploy_train_mask = np.zeros(config.MAX_STATIONS, dtype=np.int8)
        select_line_mask = np.ones(len(config.LINE_COLORS), dtype=np.int8)

        # Maska wysokiego poziomu: [0: noop, 1: manage, 2: deploy, 3: select]
        # Zaczynamy z włączonymi tylko akcjami, które są zawsze "możliwe" (nawet jeśli nic nie robią).
        # Akcje [1] i [2] włączymy dynamicznie, jeśli będą dla nich dostępne pod-akcje.
        high_level_mask = np.array([1, 0, 0, 1], dtype=np.int8)

        # --- Logika dla maski "manage_line" (Akcja 1) ---
        selected_color = config.LINE_COLORS[self.selected_line_index]
        current_line = next((line for line in self.lines if line.data.color == selected_color), None)

        if current_line is None:
            if num_current_stations >= 2:
                manage_line_type_mask[0] = 1
                for i in range(num_current_stations):
                    for j in range(num_current_stations):
                        if i != j:
                            manage_line_mask[0, i, j] = 1
        else:
            # Akcja: Usuń linię (Typ 1)
            # Zawsze można usunąć istniejącą linię.
            manage_line_type_mask[1] = 1
            possible_extensions_found = False

            if len(current_line.stations) == 1:
                if num_current_stations > 1:
                    try:
                        ep_idx = self.stations.index(current_line.stations[0])
                        for j in range(num_current_stations):
                            if j != ep_idx:
                                manage_line_mask[0, ep_idx, j] = 1
                                possible_extensions_found = True
                    except ValueError:
                        pass

            elif len(current_line.stations) > 1:
                endpoints = [current_line.stations[0], current_line.stations[-1]]
                endpoint_indices = [self.stations.index(s) for s in endpoints if s in self.stations]

                for ep_idx in endpoint_indices:
                    for j in range(num_current_stations):
                        if self.stations[j] not in current_line.stations:
                            manage_line_mask[0, ep_idx, j] = 1
                            possible_extensions_found = True

            if possible_extensions_found:
                manage_line_type_mask[0] = 1

        # --- Logika dla maski "deploy_train" (Akcja 2) ---
        if self.available_trains > 0:
            for i, station in enumerate(self.stations):
                if any(station.data.station_id in line.data.station_ids for line in self.lines):
                    deploy_train_mask[i] = 1

        # --- Finalizacja maski wysokiego poziomu ---

        if np.any(manage_line_type_mask):
            high_level_mask[1] = 1

        if np.any(deploy_train_mask):
            high_level_mask[2] = 1

        return {
            "high_level": high_level_mask,
            "manage_line": manage_line_mask,
            "deploy_train": deploy_train_mask,
            "select_line": select_line_mask,
            "manage_line_type": manage_line_type_mask
        }

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.screen = self.clock = self.font = self.font_small = self.font_large = None

        max_idx_for_params = max(config.MAX_STATIONS, len(config.LINE_COLORS))
        self.action_space = spaces.Dict({
            "high_level_action": spaces.Discrete(4),
            "low_level_params": spaces.MultiDiscrete([3, max_idx_for_params, max_idx_for_params])
        })

        NUM_NODE_FEATURES = (
                2 +  # Pozycja (x, y)
                len(config.STATION_SHAPES) +  # Kształt stacji (one-hot)
                len(config.STATION_SHAPES) +  # Popyt pasażerów (wektor dla każdego kształtu)
                1  # Czy stacja jest przepełniona
        )

        # Obliczamy liczbę cech dla każdej krawędzi (połączenia)
        NUM_EDGE_FEATURES = (
                len(config.LINE_COLORS) +  # Kolor linii (one-hot)
                1  # Znormalizowana odległość
        )

        # Definiujemy maksymalną możliwą liczbę krawędzi w grafie (w obie strony)
        MAX_EDGES = config.MAX_STATIONS * (config.MAX_STATIONS - 1)


        self.observation_space = spaces.Dict({
            # Cechy globalne całej gry (wynik, tydzień, pociągi, czas)
            "global_features": spaces.Box(
                low=0, high=np.inf, shape=(4,), dtype=np.float32
            ),

            # Tabela cech wszystkich węzłów (stacji)
            "node_features": spaces.Box(
                low=0, high=1.0, shape=(config.MAX_STATIONS, NUM_NODE_FEATURES), dtype=np.float32
            ),

            # Lista krawędzi w formacie [source_nodes, destination_nodes]
            "edge_index": spaces.Box(
                low=0, high=config.MAX_STATIONS - 1, shape=(2, MAX_EDGES), dtype=np.int32
            ),

            # Tabela cech wszystkich krawędzi (kolor + odległość)
            "edge_features": spaces.Box(
                low=0, high=1.0, shape=(MAX_EDGES, NUM_EDGE_FEATURES), dtype=np.float32
            ),
            "action_masks": spaces.Dict({
                "high_level": spaces.Box(low=0, high=1, shape=(4,), dtype=np.int8),
                "manage_line": spaces.Box(low=0, high=1, shape=(3, config.MAX_STATIONS, config.MAX_STATIONS),
                                          dtype=np.int8),
                "deploy_train": spaces.Box(low=0, high=1, shape=(config.MAX_STATIONS,), dtype=np.int8),
                "select_line": spaces.Box(low=0, high=1, shape=(len(config.LINE_COLORS),), dtype=np.int8)
            }),

            "num_edges": spaces.Box(low=0, high=MAX_EDGES, shape=(1,), dtype=np.int32
            ),

            "num_nodes": spaces.Box(low=0, high=config.MAX_STATIONS, shape=(1,), dtype=np.int32
            ),

        })

    def _get_obs(self) -> dict:
        station_id_to_idx = {s.data.station_id: i for i, s in enumerate(self.stations)}

        num_current_stations = len(self.stations)

        global_features = np.array([
            self.score,
            self.week_number,
            self.available_trains,
            self.week_timer / config.WEEK_DURATION_FRAMES
        ], dtype=np.float32)

        node_features = np.zeros(self.observation_space["node_features"].shape, dtype=np.float32)
        for i, station in enumerate(self.stations):
            pos_x = station.pos.x / config.SCREEN_WIDTH
            pos_y = station.pos.y / config.SCREEN_HEIGHT
            shape_idx = config.STATION_SHAPES.index(station.data.shape)
            shape_vec = np.eye(len(config.STATION_SHAPES))[shape_idx]
            passenger_demand = np.zeros(len(config.STATION_SHAPES), dtype=np.float32)
            for p_model in station.data.passengers:
                target_shape_idx = config.STATION_SHAPES.index(p_model.target_shape)
                passenger_demand[target_shape_idx] += 1
            passenger_demand /= config.STATION_CAPACITY
            is_overcrowded = 1.0 if station.data.is_overcrowded else 0.0
            node_features[i] = np.concatenate([
                np.array([pos_x, pos_y]), shape_vec, passenger_demand, np.array([is_overcrowded])
            ])

        edge_list = []
        edge_feature_list = []

        for u_id, v_id, key, edge_data in self.G.edges(data=True, keys=True):
            if u_id in station_id_to_idx and v_id in station_id_to_idx:
                u, v = station_id_to_idx[u_id], station_id_to_idx[v_id]
                u_station, v_station = self.stations[u], self.stations[v]
                distance = u_station.pos.distance_to(v_station.pos) / config.SCREEN_WIDTH
                color = edge_data.get('color', config.GRAY)
                color_vec = np.zeros(len(config.LINE_COLORS), dtype=np.float32)
                if color in config.LINE_COLORS:
                    color_vec[config.LINE_COLORS.index(color)] = 1.0

                edge_feature = np.concatenate([color_vec, [distance]])
                edge_list.append([u, v])
                edge_feature_list.append(edge_feature)
                edge_list.append([v, u])
                edge_feature_list.append(edge_feature)

        num_edges = len(edge_list)
        edge_index = np.zeros(self.observation_space["edge_index"].shape, dtype=np.int32)
        edge_features = np.zeros(self.observation_space["edge_features"].shape, dtype=np.float32)
        if num_edges > 0:
            if num_edges > self.observation_space["edge_index"].shape[1]:
                edge_list = edge_list[:self.observation_space["edge_index"].shape[1]]
                edge_feature_list = edge_feature_list[:self.observation_space["edge_index"].shape[1]]
                num_edges = self.observation_space["edge_index"].shape[1]

            edge_index[:, :num_edges] = np.array(edge_list).T
            edge_features[:num_edges] = np.array(edge_feature_list)

        return {
            "global_features": global_features,
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_features": edge_features,
            "action_masks": self._get_action_masks(),
            "num_edges": np.array([num_edges], dtype=np.int32),
            "num_nodes": np.array([num_current_stations], dtype=np.int32),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.screen is None and self.render_mode == "human":
            self._initialize_pygame()

        self.score = 0
        self.week_number = 1
        self.week_timer = 0
        self.available_trains = 3
        self.game_over = False

        self.station_id_counter = 0
        self.train_id_counter = 0

        self.stations: List[Station] = []
        self.lines: List[Line] = []
        self.trains: List[Train] = []

        self.G = nx.MultiGraph()
        self.passenger_spawn_timer = 0
        self.spawned_stations_this_week = 0

        self.selected_line_index = 0
        self.ui_circles_rects: List[pygame.Rect] = []

        self._generate_initial_stations()
        self.all_passengers_plan_update()

        self.deploy_train_mode = False

        obs = self._get_obs()
        info = self._get_info()
        info["action_masks"] = obs["action_masks"]

        return obs, info

    def step(self, action):
        reward = 0.0
        if not self.game_over:
            high_level_action, params = action["high_level_action"], action["low_level_params"]
            action_handlers = {1: self._handle_manage_line, 2: self._handle_deploy_train, 3: self._handle_select_line}
            if high_level_action in action_handlers:
                reward += action_handlers[high_level_action](params)

            self._update_passenger_spawning()
            reward += self._update_trains()
            self._update_station_spawning()
            reward += self._update_week_timer()

        obs = self._get_obs()
        info = self._get_info()
        info["action_masks"] = obs["action_masks"]

        return obs, reward, self.game_over, False, info

    def render(self):
        if self.screen is None and self.render_mode == "human": self._initialize_pygame()
        if self.render_mode != "human": return

        self.screen.fill(config.BLACK)
        for line in self.lines: line.draw(self.screen)
        for station in self.stations: station.draw(self.screen)
        for train in self.trains: train.draw(self.screen)
        self._draw_ui()
        if self.game_over: self._draw_game_over()

        pygame.display.flip()
        self.clock.tick(config.FPS)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None