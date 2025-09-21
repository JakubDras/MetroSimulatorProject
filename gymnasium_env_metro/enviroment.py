import pygame
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import networkx as nx
from typing import List, Dict

# Import modułów z naszego projektu
import config
from data_models import StationModel, LineModel, TrainModel, PassengerModel
from entities import Station, Line, Train


class MiniMetroEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    # --- Prywatne Metody Pomocnicze ---

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
        initial_shapes = random.sample(config.STATION_SHAPES, k=min(3, len(config.STATION_SHAPES)))
        while len(self.stations) < 3:
            x = random.randint(int(config.SCREEN_WIDTH * 0.2), int(config.SCREEN_WIDTH * 0.8))
            y = random.randint(int(config.SCREEN_HEIGHT * 0.2), int(config.SCREEN_HEIGHT * 0.8))

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

    def _handle_manage_line(self, params) -> float:
        line_action_type, p1, p2 = params
        selected_color = config.LINE_COLORS[self.selected_line_index]
        current_line = next((line for line in self.lines if line.data.color == selected_color), None)

        if line_action_type == 0:  # Połącz/przedłuż stacje
            if not (0 <= p1 < len(self.stations) and 0 <= p2 < len(self.stations) and p1 != p2):
                return -0.1

            s1, s2 = self.stations[p1], self.stations[p2]
            if current_line is None:
                if self.available_trains <= 0: return -0.2

                line_model = LineModel(color=selected_color, station_ids=[s1.data.station_id, s2.data.station_id])
                new_line = Line(line_model, self.stations)
                self.lines.append(new_line)

                train_model = TrainModel(
                    train_id=self._get_new_train_id(),
                    line_color=selected_color,
                    current_station_id=s1.data.station_id,
                    target_station_id=s2.data.station_id,
                    pos=s1.data.pos
                )
                self.trains.append(Train(train_model, new_line))
                self.available_trains -= 1
                return 0.5
            else:
                if current_line.stations[-1] == s1 and current_line.add_station(s2, self.stations):
                    return 0.1
                elif current_line.stations[0] == s1 and current_line.add_station(s2, self.stations, index=0):
                    return 0.1

        elif line_action_type == 1:  # Usuń linię
            if current_line:
                trains_to_keep = [t for t in self.trains if t.line != current_line]
                self.available_trains += len(self.trains) - len(trains_to_keep)
                self.trains = trains_to_keep
                self.lines.remove(current_line)
                return 0.1

        return -0.05

    def _handle_deploy_train(self, params) -> float:
        station_idx = params[1]
        if self.available_trains > 0 and 0 <= station_idx < len(self.stations):
            target_station = self.stations[station_idx]
            for line in self.lines:
                if target_station in line.stations and len(line.stations) >= 2:
                    next_id = line.get_next_station_id(target_station.data.station_id, 1)
                    if not next_id:
                        next_id = line.get_next_station_id(target_station.data.station_id, -1)

                    if next_id:
                        train_model = TrainModel(
                            train_id=self._get_new_train_id(),
                            line_color=line.data.color,
                            current_station_id=target_station.data.station_id,
                            target_station_id=next_id,
                            pos=target_station.data.pos
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
        self.passenger_spawn_timer += self.clock.get_time()
        if self.passenger_spawn_timer >= config.PASSENGER_SPAWN_RATE:
            self.passenger_spawn_timer = 0
            station = random.choice(self.stations)

            if len(station.data.passengers) < station.capacity:
                possible_targets = [s.data.shape for s in self.stations if s.data.shape != station.data.shape]
                if not possible_targets: return

                target_shape = random.choice(possible_targets)
                p_model = PassengerModel(origin_station_id=station.data.station_id, target_shape=target_shape)
                self.travel_planner_for_new_passager(p_model)
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

        for train in self.trains:
            if train.data.passengers:
                reward += 0.001
        return reward

    def _update_station_spawning(self):
        # ... Logika spawnowania nowych stacji (jak w oryginale, ale używając nowego tworzenia obiektów) ...
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
        # ... Logika rysowania UI (jak w oryginale) ...
        pass

    def _draw_game_over(self):
        # ... Logika rysowania ekranu Game Over (jak w oryginale) ...
        pass

    def travel_planner_for_new_passager(self, passenger_model: PassengerModel):
        # Działa na PassengerModel
        pass

    def all_passengers_plan_update(self):
        # Iteruje po s.data.passengers (modelach)
        pass



    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None
        self.font_small = None

        # Definicja przestrzeni akcji i obserwacji
        max_idx_for_params = max(config.MAX_STATIONS, len(config.LINE_COLORS))
        self.action_space = spaces.Dict({
            "high_level_action": spaces.Discrete(4),  # 0:NOOP, 1:MANAGE_LINE, 2:DEPLOY_TRAIN, 3:SELECT_LINE
            "low_level_params": spaces.MultiDiscrete([
                3,  # Typ pod-akcji dla MANAGE_LINE
                max_idx_for_params,
                max_idx_for_params
            ])
        })

        self.observation_space = spaces.Dict({
            "game_state": spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32),
            "stations_data": spaces.Box(low=0, high=1, shape=(config.MAX_STATIONS,
                                                              2 + 1 + 1 + len(config.STATION_SHAPES) + len(
                                                                  config.LINE_COLORS)), dtype=np.float32),
            "passengers_on_stations_targets": spaces.Box(low=0, high=1,
                                                         shape=(config.MAX_STATIONS, config.STATION_CAPACITY,
                                                                len(config.STATION_SHAPES)), dtype=np.int8),
            "lines_data": spaces.Box(low=0, high=1,
                                     shape=(len(config.LINE_COLORS), 2 + config.MAX_STATIONS * config.MAX_STATIONS),
                                     dtype=np.int8),
            "trains_data": spaces.Box(low=0, high=1, shape=(config.MAX_TRAINS, 2 + 2 + 1 + 1), dtype=np.float32),
            "passengers_in_trains_targets": spaces.Box(low=0, high=1, shape=(config.MAX_TRAINS, config.TRAIN_CAPACITY,
                                                                             len(config.STATION_SHAPES)),
                                                       dtype=np.int8),
        })

    def _get_obs(self) -> Dict:
        # Ta metoda musi być w pełni zaimplementowana,
        # odwołując się do `obj.data` w celu uzyskania stanu.
        # Przykład dla stanu gry:
        game_state = np.array([self.score, self.week_number, self.available_trains, self.week_timer], dtype=np.float32)

        # Poniżej placeholder - wymaga pełnej implementacji
        return {
            "game_state": game_state,
            "stations_data": np.zeros_like(self.observation_space["stations_data"].low),
            "passengers_on_stations_targets": np.zeros_like(
                self.observation_space["passengers_on_stations_targets"].low),
            "lines_data": np.zeros_like(self.observation_space["lines_data"].low),
            "trains_data": np.zeros_like(self.observation_space["trains_data"].low),
            "passengers_in_trains_targets": np.zeros_like(self.observation_space["passengers_in_trains_targets"].low),
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

        self.G = nx.Graph()
        self.passenger_spawn_timer = 0
        self.spawned_stations_this_week = 0

        self.selected_line_index = 0
        self.ui_circles_rects: List[pygame.Rect] = []

        self._generate_initial_stations()
        self.all_passengers_plan_update()

        return self._get_obs(), self._get_info()

    def step(self, action):
        reward = 0.0

        if not self.game_over:
            high_level_action = action["high_level_action"]
            params = action["low_level_params"]

            if high_level_action == 1:
                reward += self._handle_manage_line(params)
            elif high_level_action == 2:
                reward += self._handle_deploy_train(params)
            elif high_level_action == 3:
                reward += self._handle_select_line(params)

            self._update_passenger_spawning()
            reward += self._update_trains()
            self._update_station_spawning()
            reward += self._update_week_timer()

        terminated = self.game_over
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def render(self):
        if self.screen is None and self.render_mode == "human":
            self._initialize_pygame()

        if self.render_mode != "human": return

        self.screen.fill(config.BLACK)

        for line in self.lines: line.draw(self.screen)
        for station in self.stations: station.draw(self.screen)
        for train in self.trains: train.draw(self.screen)

        self._draw_ui()

        if self.game_over:
            self._draw_game_over()

        pygame.display.flip()
        self.clock.tick(config.FPS)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None