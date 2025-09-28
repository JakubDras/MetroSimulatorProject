import pygame
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import networkx as nx
from typing import List

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
            if current_line is None:  # Tworzenie nowej linii
                if self.available_trains <= 0: return -0.2

                line_model = LineModel(color=selected_color, station_ids=[s1.data.station_id, s2.data.station_id])
                new_line = Line(line_model)
                self.lines.append(new_line)

                # POPRAWKA: Dodaj krawędź do grafu NetworkX
                self.G.add_edge(s1.data.station_id, s2.data.station_id, color=selected_color)

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
            else:  # Rozszerzanie istniejącej linii
                if current_line.stations[-1] == s1 and current_line.add_station(s2, self.stations):
                    # POPRAWKA: Dodaj krawędź do grafu NetworkX
                    self.G.add_edge(s1.data.station_id, s2.data.station_id, color=selected_color)
                    return 0.1
                elif current_line.stations[0] == s2 and current_line.add_station(s1, self.stations):
                    # POPRAWKA: Dodaj krawędź do grafu NetworkX
                    self.G.add_edge(s1.data.station_id, s2.data.station_id, color=selected_color)
                    return 0.1

        elif line_action_type == 1:  # Usuń linię
            if current_line:
                # POPRAWKA: Znajdź i usuń wszystkie krawędzie powiązane z tą linią z grafu
                edges_to_remove = [
                    (u, v) for u, v, data in self.G.edges(data=True)
                    if data.get('color') == selected_color
                ]
                self.G.remove_edges_from(edges_to_remove)

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

        for train in self.trains:
            if train.data.passengers:
                reward += 0.001
        return reward

    def _update_station_spawning(self):
        # Warunek 1: Czy w tym tygodniu mamy jeszcze stacje do wygenerowania?
        if self.spawned_stations_this_week >= config.STATIONS_TO_SPAWN_PER_WEEK:
            return

        # Warunek 2: Czy nadszedł czas na wygenerowanie kolejnej stacji?
        time_for_next_spawn = config.STATION_SPAWN_TIMES[self.spawned_stations_this_week]
        if self.week_timer < time_for_next_spawn:
            return

        spawned = False
        attempts = 0
        max_attempts = 100

        # Pętla próbująca znaleźć odpowiednie miejsce na nową stację
        while not spawned and attempts < max_attempts and len(self.stations) < config.MAX_STATIONS:
            # Wybierz losową istniejącą stację, w pobliżu której pojawi się nowa
            if not self.stations:  # Zabezpieczenie na wypadek, gdyby lista była pusta
                return
            neighbour_station = random.choice(self.stations)

            # Wygeneruj losowe koordynaty w pobliżu wybranej stacji
            x = random.randint(
                max(20, int(neighbour_station.pos.x - config.SCREEN_WIDTH * 0.15)),
                min(int(config.SCREEN_WIDTH - 20), int(neighbour_station.pos.x + config.SCREEN_WIDTH * 0.15))
            )
            y = random.randint(
                max(20, int(neighbour_station.pos.y - config.SCREEN_HEIGHT * 0.15)),
                min(int(config.SCREEN_HEIGHT - 20), int(neighbour_station.pos.y + config.SCREEN_HEIGHT * 0.15))
            )

            new_pos_vec = pygame.Vector2(x, y)

            # Sprawdź, czy nowa pozycja jest wystarczająco daleko od wszystkich istniejących stacji
            if all(new_pos_vec.distance_to(s.pos) > config.STATION_MIN_DISTANCE for s in self.stations):
                shape = random.choice(config.STATION_SHAPES)

                # --- NOWY, POPRAWNY SPOSÓB TWORZENIA OBIEKTU ---
                # 1. Stwórz model danych Pydantic
                station_model = StationModel(
                    station_id=self._get_new_station_id(),
                    pos=(x, y),  # Zapisujemy jako krotkę
                    shape=shape
                )
                # 2. Stwórz obiekt logiczny na podstawie modelu
                new_station = Station(station_model)

                # Dodaj nową stację do stanu gry
                self.stations.append(new_station)
                self.G.add_node(station_model.station_id, pos=station_model.pos)
                self.all_passengers_plan_update()  # Zaktualizuj plany podróży dla wszystkich pasażerów

                spawned = True
                self.spawned_stations_this_week += 1

            attempts += 1

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
        # === SEKCJA 1: Rysowanie kółek wyboru linii (dawne draw_ui_circles) ===
        self.ui_circles_rects = []
        start_x = 20
        y = 30
        radius = 15
        spacing = 50
        for i in range(len(config.LINE_COLORS)):
            color = config.LINE_COLORS[i]
            # Sprawdzamy, czy linia o danym kolorze istnieje, odwołując się do modelu danych
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

        # === SEKCJA 2: Rysowanie tekstu (Wynik, Tydzień) ===
        score_text = self.font.render(f"Score: {self.score}", True, config.WHITE)
        self.screen.blit(score_text, (config.SCREEN_WIDTH - 180, 10))
        week_text = self.font.render(f"Week: {self.week_number}", True, config.WHITE)
        self.screen.blit(week_text, (config.SCREEN_WIDTH - 180, 40))

        # === SEKCJA 3: Rysowanie paska postępu tygodnia ===
        pygame.draw.rect(self.screen, config.WEEK_BAR_BG_COLOR, config.WEEK_BAR_RECT)
        progress = self.week_timer / config.WEEK_DURATION_FRAMES
        current_bar_width = int(config.WEEK_BAR_RECT.width * progress)
        current_bar_rect = pygame.Rect(
            config.WEEK_BAR_RECT.left,
            config.WEEK_BAR_RECT.top,
            current_bar_width,
            config.WEEK_BAR_RECT.height
        )
        pygame.draw.rect(self.screen, config.WEEK_BAR_COLOR, current_bar_rect)
        pygame.draw.rect(self.screen, config.WHITE, config.WEEK_BAR_RECT, 1)

        # === SEKCJA 4: Rysowanie przycisku "Train" ===
        # Zakładając, że `self.deploy_train_mode` jest atrybutem klasy (np. self.deploy_train_mode = False w reset())
        button_color = config.GREEN if hasattr(self, 'deploy_train_mode') and self.deploy_train_mode else config.BLUE
        pygame.draw.rect(self.screen, button_color, config.TRAIN_BUTTON_RECT)
        pygame.draw.rect(self.screen, config.WHITE, config.TRAIN_BUTTON_RECT, 3)
        train_button_text = self.font_small.render(f"Train ({self.available_trains})", True, config.WHITE)
        text_rect = train_button_text.get_rect(center=config.TRAIN_BUTTON_RECT.center)
        self.screen.blit(train_button_text, text_rect)

    def _draw_game_over(self):
        # Utwórz półprzezroczystą, czarną warstwę, aby przyciemnić tło gry
        overlay = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))  # Ciemny kolor z 180/255 przezroczystości
        self.screen.blit(overlay, (0, 0))

        # Przygotuj teksty do wyświetlenia
        game_over_text = self.font_large.render("Game Over", True, config.RED)
        score_text = self.font.render(f"Final Score: {self.score}", True, config.WHITE)
        week_text = self.font.render(f"Reached Week: {self.week_number}", True, config.WHITE)

        # Wyśrodkuj teksty na ekranie
        game_over_rect = game_over_text.get_rect(
            center=(config.SCREEN_WIDTH / 2, config.SCREEN_HEIGHT / 2 - 40)
        )
        score_rect = score_text.get_rect(
            center=(config.SCREEN_WIDTH / 2, config.SCREEN_HEIGHT / 2 + 20)
        )
        week_rect = week_text.get_rect(
            center=(config.SCREEN_WIDTH / 2, config.SCREEN_HEIGHT / 2 + 50)
        )

        # Narysuj teksty na ekranie
        self.screen.blit(game_over_text, game_over_rect)
        self.screen.blit(score_text, score_rect)
        self.screen.blit(week_text, week_rect)
    def travel_planner_for_new_passenger(self, passenger_model: PassengerModel):
        """
            Znajduje najkrótszą ścieżkę dla nowego pasażera i aktualizuje jego listę podróży.
            Metoda operuje bezpośrednio na modelu danych Pydantic.
            """
        final_destination_station = None
        final_shortest_distance = float('inf')
        path_found = False

        try:
            # 1. Znajdź wszystkie stacje, które pasują do docelowego kształtu pasażera.
            #    Odwołujemy się do `s.data.shape`, aby uzyskać kształt z modelu stacji.
            target_stations = [s for s in self.stations if s.data.shape == passenger_model.target_shape]

            # 2. Spośród wszystkich możliwych celów znajdź ten, do którego ścieżka w grafie jest najkrótsza.
            for station in target_stations:
                try:
                    # Oblicz długość ścieżki w grafie NetworkX.
                    # Używamy ID stacji z modeli: passenger_model.origin_station_id i station.data.station_id.
                    distance = nx.shortest_path_length(
                        self.G,
                        source=passenger_model.origin_station_id,
                        target=station.data.station_id,
                        weight='weight'
                    )

                    if distance < final_shortest_distance:
                        final_shortest_distance = distance
                        final_destination_station = station
                        path_found = True

                except nx.NetworkXNoPath:
                    # Jeśli do tej konkretnej stacji nie ma ścieżki, zignoruj ją.
                    continue

            # 3. Jeśli znaleziono osiągalny cel, oblicz pełną ścieżkę i zapisz ją w modelu pasażera.
            if path_found and final_destination_station:
                path = nx.dijkstra_path(
                    self.G,
                    source=passenger_model.origin_station_id,
                    target=final_destination_station.data.station_id,
                    weight='weight'
                )
                # Zapisz listę ID stacji do odwiedzenia (bez stacji początkowej, stąd [1:])
                passenger_model.travel_list = path[1:]
            else:
                # Jeśli nie znaleziono żadnej ścieżki, lista podróży pozostaje pusta.
                passenger_model.travel_list = []

        except Exception:
            # Ogólne zabezpieczenie na wypadek nieoczekiwanych błędów, np. gdy graf jest pusty.
            passenger_model.travel_list = []

    def all_passengers_plan_update(self):
        """
            Przelicza i aktualizuje plany podróży dla WSZYSTKICH pasażerów
            oczekujących na WSZYSTKICH stacjach. Przydatne po zmianie w sieci.
            """
        # Iteruj po wszystkich stacjach w grze
        for station_obj in self.stations:
            # Iteruj po modelach pasażerów na danej stacji
            for p_model in station_obj.data.passengers:
                final_destination_station = None
                final_shortest_distance = float('inf')
                path_found = False

                try:
                    # 1. Znajdź wszystkie możliwe stacje docelowe
                    target_stations = [s for s in self.stations if s.data.shape == p_model.target_shape]

                    # 2. Znajdź najbliższą z nich
                    for target in target_stations:
                        try:
                            # Stacją startową jest stacja, na której aktualnie jest pasażer
                            source_id = station_obj.data.station_id
                            target_id = target.data.station_id

                            distance = nx.shortest_path_length(
                                self.G,
                                source=source_id,
                                target=target_id,
                                weight='weight'
                            )

                            if distance < final_shortest_distance:
                                final_shortest_distance = distance
                                final_destination_station = target
                                path_found = True
                            # Dodatkowa logika z oryginału: jeśli dystans jest taki sam, wybierz losowo
                            elif distance == final_shortest_distance:
                                final_destination_station = random.choice([final_destination_station, target])

                        except nx.NetworkXNoPath:
                            continue

                    # 3. Zaktualizuj plan podróży w modelu pasażera
                    if path_found and final_destination_station:
                        path = nx.dijkstra_path(
                            self.G,
                            source=station_obj.data.station_id,
                            target=final_destination_station.data.station_id,
                            weight='weight'
                        )
                        p_model.travel_list = path[1:]
                    else:
                        p_model.travel_list = []

                except Exception as e:
                    print(f"Błąd podczas aktualizacji planu pasażera: {e}")
                    p_model.travel_list = []

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.screen = None
        self.clock = None
        self.font = None
        self.font_small = None
        self.font_large = None

        max_idx_for_params = max(config.MAX_STATIONS, len(config.LINE_COLORS))
        self.action_space = spaces.Dict({
            "high_level_action": spaces.Discrete(4),
            "low_level_params": spaces.MultiDiscrete([
                3,
                max_idx_for_params,
                max_idx_for_params
            ])
        })

        # ======================================================================
        # ZMIANA: NOWA PRZESTRZEŃ OBSERWACJI DLA GNN
        # ======================================================================

        # Obliczamy liczbę cech dla każdego węzła (stacji)
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

        # --- Finalna definicja przestrzeni obserwacji ---

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
            )
        })

    def _get_obs(self) -> dict:
        # --- 1. Mapowanie ID stacji na indeksy tablicy (0, 1, 2...) ---
        station_id_to_idx = {s.data.station_id: i for i, s in enumerate(self.stations)}

        # --- 2. Przygotowanie cech globalnych ---
        global_features = np.array([
            self.score,
            self.week_number,
            self.available_trains,
            self.week_timer / config.WEEK_DURATION_FRAMES
        ], dtype=np.float32)

        # --- 3. Przygotowanie cech węzłów (Stacji) ---
        node_features_shape = self.observation_space["node_features"].shape
        node_features = np.zeros(node_features_shape, dtype=np.float32)

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
                np.array([pos_x, pos_y]),
                shape_vec,
                passenger_demand,
                np.array([is_overcrowded])
            ])

        # --- 4. Przygotowanie listy krawędzi i ich cech (z networkx) ---
        edge_list = []
        edge_feature_list = []

        for u_id, v_id, edge_data in self.G.edges(data=True):
            if u_id in station_id_to_idx and v_id in station_id_to_idx:
                u, v = station_id_to_idx[u_id], station_id_to_idx[v_id]

                # ZMIANA: Pobieramy obiekty stacji, aby obliczyć dystans
                u_station, v_station = self.stations[u], self.stations[v]

                # ZMIANA: Obliczamy i normalizujemy odległość
                distance = u_station.pos.distance_to(v_station.pos) / config.SCREEN_WIDTH

                # Przygotowujemy wektor koloru (one-hot)
                color = edge_data.get('color', config.GRAY)
                color_vec = np.zeros(len(config.LINE_COLORS), dtype=np.float32)
                try:
                    color_idx = config.LINE_COLORS.index(color)
                    color_vec[color_idx] = 1.0
                except ValueError:
                    pass  # Kolor nie jest kolorem linii, zostawiamy wektor zerowy

                # ZMIANA: Łączymy wektor koloru z odległością
                edge_feature = np.concatenate([color_vec, [distance]])

                # Dodajemy krawędź i jej cechy dla obu kierunków
                edge_list.append([u, v])
                edge_feature_list.append(edge_feature)
                edge_list.append([v, u])
                edge_feature_list.append(edge_feature)

        # Wypełnienie do stałego rozmiaru (padding)
        num_edges = len(edge_list)
        edge_index = np.zeros(self.observation_space["edge_index"].shape, dtype=np.int32)
        edge_features = np.zeros(self.observation_space["edge_features"].shape, dtype=np.float32)

        if num_edges > 0:
            edge_index[:, :num_edges] = np.array(edge_list).T
            edge_features[:num_edges] = np.array(edge_feature_list)

        # --- 5. Złożenie finalnej obserwacji ---
        return {
            "global_features": global_features,
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_features": edge_features
        }

    def reset(self, seed=None, options=None):
        # POPRAWKA 1: Dodajemy wywołanie super().reset() na początku
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

        # POPRAWKA 2: Używamy MultiGraph, aby zezwolić na wiele linii między stacjami
        self.G = nx.MultiGraph()
        self.passenger_spawn_timer = 0
        self.spawned_stations_this_week = 0

        self.selected_line_index = 0
        self.ui_circles_rects: List[pygame.Rect] = []

        self._generate_initial_stations()
        self.all_passengers_plan_update()

        self.deploy_train_mode = False

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