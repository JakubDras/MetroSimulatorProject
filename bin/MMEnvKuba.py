
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random
import networkx as nx
import pygame
import math



def draw_passenger_icon(surface, shape, color, center_pos, size):
    half_size = size // 2
    outline_thickness = 1

    if shape == 'circle':
        pygame.draw.circle(surface, color, center_pos, half_size)
        pygame.draw.circle(surface, (0,0,0), center_pos, half_size, outline_thickness)
    elif shape == 'square':
        rect = pygame.Rect(center_pos.x - half_size, center_pos.y - half_size, size, size)
        pygame.draw.rect(surface, color, rect)
        pygame.draw.rect(surface, (0,0,0), rect, outline_thickness)
    elif shape == 'triangle':
        points = [
            (center_pos.x, center_pos.y - half_size),
            (center_pos.x - half_size, center_pos.y + half_size * 0.8),
            (center_pos.x + half_size, center_pos.y + half_size * 0.8)]
        pygame.draw.polygon(surface, color, points)
        pygame.draw.polygon(surface, (0,0,0), points, outline_thickness)
    elif shape == 'star':
        points = [
            (center_pos.x, center_pos.y - size * 0.7),
            (center_pos.x + size * 0.16, center_pos.y - size * 0.16),
            (center_pos.x + size * 0.7, center_pos.y),
            (center_pos.x + size * 0.16, center_pos.y + size * 0.16),
            (center_pos.x, center_pos.y + size * 0.7),
            (center_pos.x - size * 0.16, center_pos.y + size * 0.16),
            (center_pos.x - size * 0.7, center_pos.y),
            (center_pos.x - size * 0.16, center_pos.y - size * 0.16)
        ]
        pygame.draw.polygon(surface, color, points)
        pygame.draw.polygon(surface, (0,0,0), points, outline_thickness)

    elif shape == 'rombus':
        points = [
            (center_pos.x, center_pos.y - size * 0.5),
            (center_pos.x + size * 0.5, center_pos.y),
            (center_pos.x, center_pos.y + size * 0.5),
            (center_pos.x - size * 0.5, center_pos.y),
        ]
        pygame.draw.polygon(surface, color, points)
        pygame.draw.polygon(surface, (0,0,0), points, outline_thickness)
class Passenger:
    def __init__(self, origin_station, target_shape):
        self.origin_station = origin_station
        self.target_shape = target_shape
        self.travel_list = []
class Station:
    def __init__(self,x, y, shape,env):
        self.station_id = env.get_station_id()
        self.pos = pygame.Vector2(x, y)
        self.shape = shape
        self.radius = env.STATION_RADIUS
        self.color = env.WHITE
        self.passengers = []
        self.capacity = env.STATION_CAPACITY
        self.is_overcrowded = False
        self.passenger_timer = 0

    def draw(self, screen,env):
        line_thickness = env.STATION_BORDER_THICKNESS + (2 if self.is_overcrowded else 0)
        outline_color = env.RED if self.is_overcrowded else self.color

        if self.shape == 'circle':
            pygame.draw.circle(screen, outline_color, self.pos, self.radius, line_thickness)
        elif self.shape == 'square':
            pygame.draw.rect(screen, outline_color, (self.pos.x - self.radius, self.pos.y - self.radius, self.radius*2, self.radius*2), line_thickness)
        elif self.shape == 'triangle':
            points = [
                (self.pos.x, self.pos.y - self.radius),
                (self.pos.x - self.radius, self.pos.y + self.radius * 0.8),
                (self.pos.x + self.radius, self.pos.y + self.radius * 0.8)]
            pygame.draw.polygon(screen, outline_color, points, line_thickness)
        elif self.shape == 'star':
             points = [
                (self.pos.x, self.pos.y - self.radius * 1.2),
                (self.pos.x + self.radius * 0.35, self.pos.y - self.radius * 0.35),
                (self.pos.x + self.radius * 1.2, self.pos.y),
                (self.pos.x + self.radius * 0.35, self.pos.y + self.radius * 0.35),
                (self.pos.x, self.pos.y + self.radius * 1.2),
                (self.pos.x - self.radius * 0.35, self.pos.y + self.radius * 0.35),
                (self.pos.x - self.radius * 1.2, self.pos.y),
                (self.pos.x - self.radius * 0.35, self.pos.y - self.radius * 0.35)
            ]
             pygame.draw.polygon(screen, outline_color, points, line_thickness)
        elif self.shape == 'rombus':
            points = [
                (self.pos.x, self.pos.y - self.radius),
                (self.pos.x + self.radius, self.pos.y),
                (self.pos.x, self.pos.y + self.radius),
                (self.pos.x - self.radius, self.pos.y),
            ]
            pygame.draw.polygon(screen, outline_color, points, line_thickness+1)


        num_passengers = len(self.passengers)
        if num_passengers > 0:
            icon_size = env.PASSENGER_ICON_SIZE
            spacing = env.PASSENGER_ICON_SPACING
            total_width = num_passengers * icon_size + max(0, num_passengers - 1) * spacing

            row_y_center = self.pos.y - self.radius - 15
            start_x_center = self.pos.x - total_width / 2 + icon_size / 2

            for i, passenger in enumerate(self.passengers):
                icon_center_x = start_x_center + i * (icon_size + spacing)
                icon_center_pos = pygame.Vector2(icon_center_x, row_y_center)
                icon_color = env.WHITE

                draw_passenger_icon(screen, passenger.target_shape, icon_color, icon_center_pos, icon_size)


    def add_passenger(self, all_stations,env):
        if len(self.passengers) >= self.capacity:
            self.is_overcrowded = True
            return False

        possible_targets = [s.shape for s in all_stations if s.shape != self.shape]
        if not possible_targets:
             if len(all_stations) > 1:
                 target_shape = random.choice([s for s in env.STATION_SHAPES if s != self.shape])
                 p = Passenger(self,target_shape)
                 env.travel_planner_for_new_passager(p)
                 self.passengers.append(p)
                 return True
             else:
                return True

        target_shape = random.choice(possible_targets)
        p = Passenger(self, target_shape)
        env.travel_planner_for_new_passager(p)
        self.passengers.append(p)

        return True

    def is_clicked(self, mouse_pos):
        return self.pos.distance_to(mouse_pos) <= self.radius + 5
class Line:
    def __init__(self, color):
        self.stations = []
        self.color = color
        self.is_loop = False

    def add_station(self, station, index=-1):
        is_closing_loop = False
        if len(self.stations) >= 2:
            if index == -1 and station == self.stations[0]:
                 is_closing_loop = True
            elif index == 0 and station == self.stations[-1]:
                 is_closing_loop = True

        if station in self.stations and not is_closing_loop:
             return False

        if index == -1:
            self.stations.append(station)
        else:
            self.stations.insert(index, station)

        self.check_loop()
        return True

    def remove_station(self, station):
        if station in self.stations:
            indices_to_remove = [i for i, s in enumerate(self.stations) if s == station]
            for i in sorted(indices_to_remove, reverse=True):
                 del self.stations[i]

            self.check_loop()
            return True
        return False

    def check_loop(self):
        self.is_loop = len(self.stations) > 2 and self.stations[0] == self.stations[-1]

    def draw(self, screen):
        if len(self.stations) >= 2:
            points = [s.pos for s in self.stations]
            pygame.draw.lines(screen, self.color, self.is_loop, points, 5)

    def has_station(self, target_id):
        return any(station.station_id == target_id for station in self.stations)
class Train:
    def __init__(self, line,env):
        if not line or len(line.stations) < 2:
            raise ValueError("Pociąg musi być przypisany do linii z co najmniej dwiema stacjami.")

        self.id = env.get_train_id()
        self.line = line
        self.color = line.color
        self.current_station_id = self.line.stations[0].station_id
        self.target_station_id = self.line.stations[1].station_id
        self.pos = pygame.Vector2(self.line.stations[0].pos)
        self.direction = 1
        self.speed = env.TRAIN_SPEED
        self.size = env.TRAIN_SIZE
        self.passengers = []
        self.capacity = env.TRAIN_CAPACITY

        self.image = pygame.Surface((self.size, self.size * 2), pygame.SRCALPHA)
        self.image.fill(self.color)
        pygame.draw.rect(self.image, (255,255,255), (0, 0, self.size, self.size * 2), 1)

    def find_station_index_by_id(self, station_id):
        for i, station in enumerate(self.line.stations):
            if station.station_id == station_id:
                return i
        return -1

    def get_station_by_id(self, station_id):
        for station in self.line.stations:
            if station.station_id == station_id:
                return station
        return None

    def update(self,env):
        if not self.line or len(self.line.stations) < 2:
            return False

        current_index = self.find_station_index_by_id(self.current_station_id)
        target_index = self.find_station_index_by_id(self.target_station_id)


        if current_index == -1 or target_index == -1:
            print("Błąd: stacja aktualna lub docelowa nie istnieje na linii. Usuwam pociąg.")
            return False

        target_station = self.line.stations[target_index]
        target_pos = target_station.pos
        move_vector = target_pos - self.pos
        distance_to_target = move_vector.length()

        if distance_to_target < self.speed:
            self.pos = pygame.Vector2(target_pos)
            self.current_station_id = self.target_station_id
            arrived_station = self.get_station_by_id(self.current_station_id)

            num_stations = len(self.line.stations)
            if num_stations < 2:
                return False

            current_index = self.find_station_index_by_id(self.current_station_id)

            if self.line.is_loop:
                next_index = (current_index + self.direction) % num_stations
                if self.direction == -1 and next_index == num_stations - 1:
                    next_index -= 1
                self.target_station_id = self.line.stations[next_index].station_id
            else:
                next_index = current_index + self.direction
                if 0 <= next_index < num_stations:
                    self.target_station_id = self.line.stations[next_index].station_id
                else:
                    self.direction *= -1
                    next_index = current_index + self.direction
                    if 0 <= next_index < num_stations:
                        self.target_station_id = self.line.stations[next_index].station_id
                    else:
                        print("Nie udało się znaleźć nowej stacji, resetuję pociąg.")
                        self.current_station_id = self.line.stations[0].station_id
                        self.target_station_id = self.line.stations[1 % num_stations].station_id
                        self.direction = 1
                        self.pos = pygame.Vector2(self.line.stations[0].pos)
                        return False

            passengers_staying = []
            for p in self.passengers:
                if p.target_shape == arrived_station.shape:
                    env.score += 1
                elif len(p.travel_list) == 0:
                    arrived_station.passengers.append(p)
                    env.travel_planner_for_new_passager(p)
                elif p.travel_list[0] == arrived_station.station_id:
                    p.travel_list.pop(0)
                    arrived_station.passengers.append(p)
                else:
                    passengers_staying.append(p)
            self.passengers = passengers_staying

            if not arrived_station.is_overcrowded:
                boarding_list = list(arrived_station.passengers)
                random.shuffle(boarding_list)

                for p in boarding_list:
                    if p.travel_list:
                        if len(self.passengers) < self.capacity and self.line.has_station(p.travel_list[0]) and p.travel_list[0] == self.target_station_id:
                            self.passengers.append(p)

                            if p in arrived_station.passengers:
                                arrived_station.passengers.remove(p)

            if arrived_station.is_overcrowded and len(arrived_station.passengers) < arrived_station.capacity:
                arrived_station.is_overcrowded = False

        else:
            if distance_to_target > 0:
                try:
                    move_vector.normalize_ip()
                    self.pos += move_vector * self.speed
                except ValueError:
                    self.pos = pygame.Vector2(target_pos)
            else:
                self.pos = pygame.Vector2(target_pos)

        return True

    def draw(self, screen):
        if len(self.line.stations) < 2:
            return
        target_station = self.get_station_by_id(self.target_station_id)
        if not target_station:
            return
        move_direction = target_station.pos - self.pos
        if move_direction.length_squared() > 0.1:
            angle = pygame.math.Vector2(0, -1).angle_to(move_direction)
            rotated_image = pygame.transform.rotate(self.image, angle)
            rect = rotated_image.get_rect(center=self.pos)
            screen.blit(rotated_image, rect)
        else:
            rect = self.image.get_rect(center=self.pos)
            screen.blit(self.image, rect)
def euclidean_distance(start_station, end_station):
    pos1 = (start_station.pos.x,start_station.pos.y)
    pos2 = (end_station.pos.x,end_station.pos.y)
    return round(math.dist(pos1,pos2))
def calculate_edge_weight(graph, node1_id, node2_id):
    pos1 = graph.nodes[node1_id]['pos']
    pos2 = graph.nodes[node2_id]['pos']
    return round(math.dist(pos1,pos2))
ui_circles_rects = []

class MiniMetroEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    SCREEN_WIDTH = 900
    SCREEN_HEIGHT = 600
    FPS = 60
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    PINK = (255, 192, 203)
    GRAY = (100, 100, 100)
    GREEN_BLUE = (50, 166, 154)
    LIGHT_GRAY = (200, 200, 200)
    LINE_COLORS = [RED, GREEN, BLUE, YELLOW, PINK, GREEN_BLUE]
    STATION_RADIUS = 9
    STATION_BORDER_THICKNESS = 3
    STATION_CAPACITY = 10
    STATION_MIN_DISTANCE = 70
    STATION_SHAPES = ['circle', 'square', 'triangle', 'star', 'rombus']
    MAX_STATIONS = 40
    MAX_TRAINS = 10
    PASSENGER_SPAWN_RATE = 6000
    PASSENGER_ICON_SIZE = 10
    PASSENGER_ICON_SPACING = 5
    TRAIN_SPEED = 1.2
    TRAIN_CAPACITY = 5
    TRAIN_SIZE = 10

    # --- Game State ---
    WEEK_DURATION = FPS * 60
    TRAIN_BUTTON_RECT = pygame.Rect(SCREEN_WIDTH - 120, 70, 100, 40)
    # --- NEW STATIONS ---
    stations_to_spawn_this_week = 5
    STATION_SPAWN_TIMES = [((WEEK_DURATION // 5) - 1) * 1, ((WEEK_DURATION // 5) - 1) * 2,
                           ((WEEK_DURATION // 5) - 1) * 3, ((WEEK_DURATION // 5) - 1) * 4,
                           ((WEEK_DURATION // 5) - 1) * 5]
    # --- WEEK TIMER DISPLAY ---
    WEEK_BAR_RECT = pygame.Rect(20, 60, 150, 15)
    WEEK_BAR_COLOR = YELLOW
    WEEK_BAR_BG_COLOR = GRAY

    def get_train_id(self):
        self.TRAIN_ID_NUMBER += 1
        name = f"T{self.TRAIN_ID_NUMBER}"
        return name
    def get_station_id(self):
        self.STATION_ID += 1
        name = f"S{self.STATION_ID}"
        return name
    def draw_ui_circles(self,screen, lines, selected_line_index):
        self.ui_circles_rects = []
        start_x = 20
        y = 30
        radius = 15
        spacing = 50
        for i in range(len(self.LINE_COLORS)):
            color = self.LINE_COLORS[i]
            line_exists = any(line.color == color for line in lines)
            rect = pygame.Rect(start_x + i * spacing - radius, y - radius, radius * 2, radius * 2)
            self.ui_circles_rects.append(rect)
            pygame.draw.circle(screen, color, rect.center, radius)
            if i == selected_line_index:
                pygame.draw.circle(screen, self.WHITE, rect.center, radius + 2, 3)
            elif line_exists:
                pygame.draw.circle(screen, self.WHITE, rect.center, radius, 1)
            else:
                pygame.draw.circle(screen, self.GRAY, rect.center, radius, 1)
    def draw_ui(self,screen, font, font_small, score, week_number, available_trains, deploy_train_mode, week_timer,
                week_duration):
        self.draw_ui_circles(screen, self.lines, self.selected_line_index)

        score_text = font.render(f"Score: {score}", True, self.WHITE)
        screen.blit(score_text, (self.SCREEN_WIDTH - 180, 10))
        week_text = font.render(f"Week: {week_number}", True, self.WHITE)
        screen.blit(week_text, (self.SCREEN_WIDTH - 180, 40))

        pygame.draw.rect(screen, self.WEEK_BAR_BG_COLOR, self.WEEK_BAR_RECT)
        progress = week_timer / week_duration
        current_bar_width = int(self.WEEK_BAR_RECT.width * progress)
        pygame.draw.rect(screen, self.WEEK_BAR_COLOR,
                         (self.WEEK_BAR_RECT.left, self.WEEK_BAR_RECT.top, current_bar_width, self.WEEK_BAR_RECT.height))
        pygame.draw.rect(screen, self.WHITE, self.WEEK_BAR_RECT, 1)  # Ramka paska

        button_color = self.GREEN if deploy_train_mode else self.BLUE
        pygame.draw.rect(screen, button_color, self.TRAIN_BUTTON_RECT)
        pygame.draw.rect(screen, self.WHITE, self.TRAIN_BUTTON_RECT, 3)
        train_button_text = font_small.render(f"Train ({available_trains})", True, self.WHITE)
        text_rect = train_button_text.get_rect(center=self.TRAIN_BUTTON_RECT.center)
        screen.blit(train_button_text, text_rect)
    def travel_planner_for_new_passager(self,new_passenger):
        final_destination_station = None
        final_shortest_distance_in_g = float('inf')
        found_path_for_passenger = False
        try:
            target_stations = [t for t in self.stations if t.shape == new_passenger.target_shape]

            for station in target_stations:
                try:
                    distance = nx.shortest_path_length(self.G, source=new_passenger.origin_station.station_id,
                                                       target=station.station_id, weight='weight')
                    if distance < final_shortest_distance_in_g:
                        final_destination_station = station
                        final_shortest_distance_in_g = distance
                        found_path_for_passenger = True
                except nx.NetworkXNoPath:
                    pass

            if found_path_for_passenger and final_destination_station is not None:
                path = nx.dijkstra_path(self.G, source=new_passenger.origin_station.station_id,
                                        target=final_destination_station.station_id, weight='weight')
                new_passenger.travel_list = path[1:]
            else:
                new_passenger.travel_list = []

        except Exception as e:
            pass
    def all_passengers_plan_update(self):
        for s in self.stations:
            for p in s.passengers:
                final_destination_station = None
                final_shortest_distance_in_g = float('inf')
                found_path_for_passenger = False
                try:
                    target_stations = [t for t in self.stations if t.shape == p.target_shape]
                    for station in target_stations:
                        try:
                            distance = nx.shortest_path_length(self.G, source=s.station_id, target=station.station_id,
                                                               weight='weight')
                            if distance < final_shortest_distance_in_g:
                                final_destination_station = station
                                final_shortest_distance_in_g = distance
                                found_path_for_passenger = True
                                # ====
                            elif distance == final_shortest_distance_in_g:
                                final_destination_station = random.choice([final_destination_station, station])
                                # =====
                        except nx.NetworkXNoPath:
                            pass
                    if found_path_for_passenger and final_destination_station is not None:
                        path = nx.dijkstra_path(self.G, source=s.station_id, target=final_destination_station.station_id,
                                                weight='weight')
                        p.travel_list = path[1:]
                    else:
                        p.travel_list = []
                except Exception as e:
                    print(f"Nieoczekiwany błąd podczas AllPassengersPlanUpdate dla pasażera na {s.station_id}: {e}")
                    p.travel_list = []

    def get_station_at(self,pos):
        for station in self.stations:
            if station.is_clicked(pos):
                return station
        return None
    def _generate_initial_stations(self):
        initial_stations_count = 3
        available_shapes = list(self.STATION_SHAPES)
        random.shuffle(available_shapes)
        initial_shapes = [available_shapes.pop() for _ in range(min(initial_stations_count, len(available_shapes)))]

        x = random.randint(round(self.SCREEN_WIDTH * 0.25), round(self.SCREEN_WIDTH * 0.75))
        y = random.randint(round(self.SCREEN_HEIGHT * 0.25), round(self.SCREEN_HEIGHT * 0.75))
        shape = initial_shapes.pop() if initial_shapes else random.choice(self.STATION_SHAPES)
        pos = pygame.Vector2(x, y)
        if not self.stations or all(pos.distance_to(s.pos) > self.STATION_MIN_DISTANCE for s in self.stations):
            self.stations.append(Station(x, y, shape,env=self))
            print(f"Dodano początkową stację: {shape} w ({x}, {y})")

        while len(self.stations) < initial_stations_count:
            stations_all = [s for s in self.stations]
            neighbour_station = random.choice(stations_all)
            x = random.randint(round(neighbour_station.pos.x - self.SCREEN_WIDTH * 0.15),
                               round(neighbour_station.pos.x + self.SCREEN_WIDTH * 0.15))
            y = random.randint(round(neighbour_station.pos.y - self.SCREEN_HEIGHT * 0.15),
                               round(neighbour_station.pos.y + self.SCREEN_HEIGHT * 0.15))
            while 50 > x or x > self.SCREEN_WIDTH - 50 or 50 > y or y > self.SCREEN_HEIGHT - 50:
                x = random.randint(round(neighbour_station.pos.x - self.SCREEN_WIDTH * 0.15),
                                   round(neighbour_station.pos.x + self.SCREEN_WIDTH * 0.15))
                y = random.randint(round(neighbour_station.pos.y - self.SCREEN_HEIGHT * 0.15),
                                   round(neighbour_station.pos.y + self.SCREEN_HEIGHT * 0.15))
            shape = initial_shapes.pop() if initial_shapes else random.choice(self.STATION_SHAPES)
            pos = pygame.Vector2(x, y)
            if not self.stations or all(pos.distance_to(s.pos) > self.STATION_MIN_DISTANCE for s in self.stations):
                self.stations.append(Station(x, y, shape,env=self))
                print(f"Dodano początkową stację: {shape} w ({x}, {y})")
        for s in self.stations:
            self.G.add_node(s.station_id, pos=(s.pos.x, s.pos.y))
            print(s.station_id, s.shape)

    def __init__(self,render_mode=None):
        super().__init__()

        self.render_mode = render_mode
        pygame.init()
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("MiniMetro Clone")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)
        self.font_large = pygame.font.Font(None, 72)

        self.TRAIN_ID_NUMBER = 0
        self.STATION_ID = 0

        self.ui_circles_rects = []
        self.stations = []
        self.lines = []
        self.trains = []
        self.passenger_timer = 0
        self.game_over = False
        self.G = nx.MultiGraph()

        self.score = 0
        self.week_number = 1
        self.week_timer = 0

        self.available_trains = 3
        self.deploy_train_mode = False
        self.spawned_stations_this_week_count = 0

        self.dragging = False
        self.drag_start_station = None
        self.drag_line_color = None
        self.selected_line_index = 0

        self._generate_initial_stations()

        self.observation_space = spaces.Dict({
            "game_state": spaces.Box(low=np.array([0,0,0,0]),high=np.array([np.inf, np.inf, np.inf, self.WEEK_DURATION]), dtype=np.float32),
            # [score, week_number, available_trains, week_timer]
            "stations_data": spaces.Box(low=0,high=1,shape=(self.MAX_STATIONS, 2 + 1 + 1 + len(self.STATION_SHAPES) + len(self.LINE_COLORS)),dtype=np.float32),
            # 2 (X, Y) ,1 (liczba_pasażerów), 1 (czy_przepełniona - binarna), len(STATION_SHAPES) (one-hot encoding dla kształtu) , len(self.LINE_COLORS) przynalezność do linii

            # Informacje o pasażerach na stacjach
            # Maksymalna liczba stacji * pojemność_stacji * (one-hot dla kształtu_celu)
            "passengers_on_stations_targets": spaces.Box(low=0, high=1, shape=(self.MAX_STATIONS, self.STATION_CAPACITY, len(self.STATION_SHAPES)), dtype=np.int8),

            "lines_data": spaces.Box(
                low=0,
                high=1,
                shape=(len(self.LINE_COLORS), 2 + self.MAX_STATIONS * self.MAX_STATIONS),
                dtype=np.int8),
            # 2 (kolor_linii_idx, czy_jest_pętlą - binarna)
            # MAX_STATIONS * MAX_STATIONS (macierz sąsiedztwa dla stacji na linii: 1 jeśli połączone, 0 w p.p.)

            "trains_data": spaces.Box(low=0,high=1,
                                      shape=(self.MAX_TRAINS, 2 + 2 + 1 + 1),
                                      dtype=np.float32),
            # 2 (X, Y)
            # 2 (current_station_idx, target_station_idx)
            # 1 (liczba_pasażerów)
            # 1 przynalezność do linii
            "passengers_in_trains_targets": spaces.Box(low=0, high=1, shape=(
                self.MAX_TRAINS, self.TRAIN_CAPACITY, len(self.STATION_SHAPES)), dtype=np.int8),  # <--- ZMIENIONE!
        })

        max_idx_for_params = max(self.MAX_STATIONS, len(self.LINE_COLORS))

        self.action_space = spaces.Dict({
            "high_level_action": spaces.Discrete(4),  # Liczba głównych typów akcji
            # 0: NOOP (Nic nie rób)
            # 1: MANAGE_LINE (Zarządzaj linią - buduj, rozbudowuj, usuwaj)
            # 2: DEPLOY_TRAIN (Rozmieść pociąg)
            # 3: SELECT_LINE_COLOR (Wybierz aktywny kolor linii)

            "low_level_params": spaces.MultiDiscrete([
                3,
                max_idx_for_params,  # Parametr 1 (np. index stacji, index koloru linii, typ zarządzania linią)
                max_idx_for_params  # Parametr 2 (np. druga stacja, dla połączeń)
            ])
        })

    def _get_obs(self):
        # --- 1. Game State ---
        game_state = np.array([
            self.score,
            self.week_number,
            self.available_trains,
            self.week_timer
        ], dtype=np.float32)

        # Helper do one-hot encodingu
        def one_hot(index, size):
            vec = np.zeros(size)
            if 0 <= index < size:
                vec[index] = 1
            return vec

        # --- 2. Stations Data ---
        stations_data = np.zeros((self.MAX_STATIONS, 2 + 1 + 1 + len(self.STATION_SHAPES) + len(self.LINE_COLORS)),
                                 dtype=np.float32)
        station_id_to_idx = {s.station_id: i for i, s in enumerate(self.stations)}

        for i, station in enumerate(self.stations):
            if i >= self.MAX_STATIONS: break
            # Normalizacja pozycji
            pos_x = station.pos.x / self.SCREEN_WIDTH
            pos_y = station.pos.y / self.SCREEN_HEIGHT

            passenger_count = len(station.passengers) / self.STATION_CAPACITY  # Normalizacja
            is_overcrowded = 1.0 if station.is_overcrowded else 0.0

            shape_vec = one_hot(self.STATION_SHAPES.index(station.shape), len(self.STATION_SHAPES))

            lines_vec = np.zeros(len(self.LINE_COLORS))
            for line_idx, color in enumerate(self.LINE_COLORS):
                for line in self.lines:
                    if line.color == color and station in line.stations:
                        lines_vec[line_idx] = 1
                        break

            stations_data[i, :] = np.concatenate(
                ([pos_x, pos_y, passenger_count, is_overcrowded], shape_vec, lines_vec))

        # --- 3. Passengers on Stations ---
        passengers_on_stations = np.zeros((self.MAX_STATIONS, self.STATION_CAPACITY, len(self.STATION_SHAPES)),
                                          dtype=np.int8)
        for i, station in enumerate(self.stations):
            if i >= self.MAX_STATIONS: break
            for p_idx, p in enumerate(station.passengers):
                if p_idx >= self.STATION_CAPACITY: break
                target_shape_idx = self.STATION_SHAPES.index(p.target_shape)
                passengers_on_stations[i, p_idx, :] = one_hot(target_shape_idx, len(self.STATION_SHAPES))

        # --- 4. Lines Data ---
        lines_data = np.zeros((len(self.LINE_COLORS), 2 + self.MAX_STATIONS * self.MAX_STATIONS), dtype=np.int8)
        for i, color in enumerate(self.LINE_COLORS):
            line = next((l for l in self.lines if l.color == color), None)
            if line:
                is_loop = 1 if line.is_loop else 0
                adj_matrix = np.zeros((self.MAX_STATIONS, self.MAX_STATIONS), dtype=np.int8)
                for j in range(len(line.stations) - 1):
                    s1_idx = station_id_to_idx.get(line.stations[j].station_id)
                    s2_idx = station_id_to_idx.get(line.stations[j + 1].station_id)
                    if s1_idx is not None and s2_idx is not None:
                        adj_matrix[s1_idx, s2_idx] = 1
                        adj_matrix[s2_idx, s1_idx] = 1  # Symetryczna
                lines_data[i, :] = np.concatenate(([i, is_loop], adj_matrix.flatten()))

        # --- 5. Trains Data ---
        trains_data = np.zeros((self.MAX_TRAINS, 2 + 2 + 1 + 1), dtype=np.float32)
        passengers_in_trains = np.zeros((self.MAX_TRAINS, self.TRAIN_CAPACITY, len(self.STATION_SHAPES)), dtype=np.int8)

        for i, train in enumerate(self.trains):
            if i >= self.MAX_TRAINS: break
            pos_x = train.pos.x / self.SCREEN_WIDTH
            pos_y = train.pos.y / self.SCREEN_HEIGHT

            current_station_idx = station_id_to_idx.get(train.current_station_id, -1)
            target_station_idx = station_id_to_idx.get(train.target_station_id, -1)

            passenger_count = len(train.passengers) / self.TRAIN_CAPACITY  # Normalizacja
            line_color_idx = self.LINE_COLORS.index(train.line.color)

            trains_data[i, :] = [pos_x, pos_y, current_station_idx, target_station_idx, passenger_count, line_color_idx]

            for p_idx, p in enumerate(train.passengers):
                if p_idx >= self.TRAIN_CAPACITY: break
                target_shape_idx = self.STATION_SHAPES.index(p.target_shape)
                passengers_in_trains[i, p_idx, :] = one_hot(target_shape_idx, len(self.STATION_SHAPES))

        # Złożenie obserwacji w słownik
        observation = {
            "game_state": game_state,
            "stations_data": stations_data,
            "passengers_on_stations_targets": passengers_on_stations,
            "lines_data": lines_data,
            "trains_data": trains_data,
            "passengers_in_trains_targets": passengers_in_trains
        }
        return observation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.TRAIN_ID_NUMBER = 0
        self.STATION_ID = 0

        self.score = 0
        self.week_number = 1
        self.week_timer = 0

        self.available_trains = 3
        self.deploy_train_mode = False
        self.spawned_stations_this_week_count = 0
        self.passenger_timer = 0
        self.game_over = False

        self.stations = []
        self.lines = []
        self.trains = []
        self.G = nx.MultiGraph()

        self.ui_circles_rects = []
        self.selected_line_index = 0

        self._generate_initial_stations()
        self.all_passengers_plan_update()

        observation = self._get_obs()

        info = {
            "week_number":self.week_number,
            "available_trains": self.available_trains,
            "score": self.score
        }
        return observation , info

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False
        info = {
            "week_number": self.week_number,
            "available_trains": self.available_trains,
            "score": self.score
        }

        # Rozpakowanie akcji wysokopoziomowej i parametrów niskopoziomowych
        high_level_action = action["high_level_action"]
        param_type = action["low_level_params"][0]  # Używane dla MANAGE_LINES
        param1_idx = action["low_level_params"][1]
        param2_idx = action["low_level_params"][2]

        # --- 1. Obsługa akcji agenta ---
        if not self.game_over:
            if high_level_action == 0:  # NOOP (Nic nie rób)
                pass
            elif high_level_action == 1:  # MANAGE_LINES
                # Parametr 1: typ akcji MANAGE_LINES (0: Połącz/przedłuż stacje, 1: Usuń całą linię)
                # Parametr 2: indeks pierwszej stacji / linii do modyfikacji
                # Parametr 3: indeks drugiej stacji (jeśli połączenie/przedłużenie)

                # Sprawdź, czy wybrany indeks linii jest prawidłowy
                if self.selected_line_index >= len(self.LINE_COLORS):
                    reward -= 0.1  # Mała kara za nieprawidłową akcję
                else:
                    current_line_color = self.LINE_COLORS[self.selected_line_index]
                    current_line = next((l for l in self.lines if l.color == current_line_color), None)

                    if param_type == 0:  # Połącz/przedłuż stacje (Connect/Extend)
                        # param1_idx to indeks pierwszej stacji, param2_idx to indeks drugiej stacji
                        if 0 <= param1_idx < len(self.stations) and 0 <= param2_idx < len(self.stations):
                            station1 = self.stations[param1_idx]
                            station2 = self.stations[param2_idx]

                            # Agent próbuje stworzyć nową linię lub rozbudować istniejącą
                            if current_line is None:
                                # Stwórz nową linię, jeśli nie istnieje i agent ma dostępne pociągi
                                if self.available_trains > 0:
                                    new_line = Line(current_line_color)
                                    # Dodaj stacje w odpowiedniej kolejności, aby utworzyć początek linii
                                    if station1.pos.x <= station2.pos.x:  # Prosta heurystyka do określenia kolejności początkowej
                                        new_line.add_station(station1)
                                        new_line.add_station(station2)
                                    else:
                                        new_line.add_station(station2)
                                        new_line.add_station(station1)

                                    self.lines.append(new_line)
                                    # Dodaj krawędzie do grafu NetworkX
                                    if self.G.has_node(station1.station_id) and self.G.has_node(station2.station_id):
                                        weight = euclidean_distance(station1, station2)
                                        self.G.add_edge(station1.station_id, station2.station_id, weight=weight,
                                                        line_color=current_line_color)
                                    self.available_trains -= 1  # Pierwszy pociąg na nowej linii jest "darmowy"
                                    self.trains.append(Train(new_line, self))  # Dodaj pociąg do nowej linii
                                    reward += 0.5  # Nagroda za stworzenie linii
                                else:
                                    reward -= 0.2  # Kara za próbę stworzenia linii bez dostępnych pociągów
                            else:
                                # Rozbuduj istniejącą linię
                                # Sprawdź, czy nowa stacja jest dodawana na końcu lub początku linii
                                if station1 == current_line.stations[-1] and current_line.add_station(station2):
                                    if self.G.has_node(station1.station_id) and self.G.has_node(station2.station_id):
                                        weight = euclidean_distance(station1, station2)
                                        self.G.add_edge(station1.station_id, station2.station_id, weight=weight,
                                                        line_color=current_line_color)
                                    reward += 0.1
                                elif station2 == current_line.stations[0] and current_line.add_station(station1, 0):
                                    if self.G.has_node(station1.station_id) and self.G.has_node(station2.station_id):
                                        weight = euclidean_distance(station1, station2)
                                        self.G.add_edge(station1.station_id, station2.station_id, weight=weight,
                                                        line_color=current_line_color)
                                    reward += 0.1
                                else:
                                    reward -= 0.1  # Kara za nieprawidłową rozbudowę linii (nie sąsiaduje)
                        else:
                            reward -= 0.1  # Kara za nieprawidłowe indeksy stacji
                    elif param_type == 1:  # Usuń całą linię (RemoveLine)
                        # Akcja usuwa AKTYWNĄ linię, której kolor jest self.LINE_COLORS[self.selected_line_index]
                        if current_line:
                            # Zbierz pociągi do usunięcia z linii i zwróć je do puli
                            trains_on_removed_line = []
                            for train in self.trains:
                                if train.line == current_line:
                                    trains_on_removed_line.append(train)
                                    # Wysadź pasażerów na obecnej stacji pociągu
                                    current_station_for_train = train.get_station_by_id(train.current_station_id)
                                    if current_station_for_train:
                                        for p in train.passengers:
                                            current_station_for_train.passengers.append(p)
                                        train.passengers = []  # Opróżnij pociąg z pasażerów
                                    self.available_trains += 1  # Zwróć pociąg do puli dostępnych

                            # Usuń pociągi z usuniętej linii z listy aktywnych pociągów
                            self.trains = [t for t in self.trains if t.line != current_line]

                            # Usuń linię z listy aktywnych linii
                            self.lines.remove(current_line)

                            # Usuń wszystkie krawędzie należące do tej linii z grafu NetworkX
                            edges_to_remove = []
                            for u, v, k, data in self.G.edges(data=True, keys=True):
                                if data.get('line_color') == current_line_color:
                                    edges_to_remove.append((u, v, k))
                            for u, v, k in edges_to_remove:
                                self.G.remove_edge(u, v, k)

                            reward += 0.1  # Nagroda za usunięcie linii
                        else:
                            reward -= 0.05  # Kara za próbę usunięcia nieistniejącej (nieaktywnej) linii

            elif high_level_action == 2:  # DEPLOY_TRAIN (Rozmieść pociąg)
                # param1_idx to indeks stacji, na której agent chce rozmieścić pociąg
                if self.available_trains > 0 and 0 <= param1_idx < len(self.stations):
                    target_station = self.stations[param1_idx]
                    # Sprawdź, czy stacja należy do jakiejś linii z co najmniej 2 stacjami
                    deployable_line = None
                    for line in self.lines:
                        if len(line.stations) >= 2 and target_station in line.stations:
                            # Sprawdź, czy stacja jest początkiem lub końcem linii (lub dowolnym punktem dla pętli)
                            if target_station == line.stations[0] or \
                                    target_station == line.stations[-1] or \
                                    (line.is_loop and target_station in line.stations):
                                deployable_line = line
                                break
                    if deployable_line:
                        new_train = Train(deployable_line, self)
                        # Ustaw początkową pozycję pociągu na wybranej stacji
                        new_train.current_station_id = target_station.station_id
                        # Znajdź następną stację na linii
                        current_idx_on_line = new_train.find_station_index_by_id(target_station.station_id)
                        if deployable_line.is_loop:
                            next_idx = (current_idx_on_line + 1) % len(deployable_line.stations)
                        else:
                            next_idx = current_idx_on_line + 1 if current_idx_on_line < len(
                                deployable_line.stations) - 1 else current_idx_on_line - 1

                        if 0 <= next_idx < len(deployable_line.stations):
                            new_train.target_station_id = deployable_line.stations[next_idx].station_id
                            new_train.pos = pygame.Vector2(target_station.pos)  # Ustaw pozycję pociągu na stacji
                            self.trains.append(new_train)
                            self.available_trains -= 1
                            reward += 0.2  # Nagroda za rozmieszczenie pociągu
                        else:
                            reward -= 0.1  # Kara za nieudane określenie następnej stacji dla pociągu
                    else:
                        reward -= 0.1  # Kara za próbę rozmieszczenia pociągu na nieodpowiedniej stacji/linii
                else:
                    reward -= 0.1  # Kara za brak dostępnych pociągów lub nieprawidłowy indeks stacji
            elif high_level_action == 3:  # SELECT_LINE_COLOR (Wybierz aktywny kolor linii)
                # param1_idx to indeks koloru linii do wybrania
                if 0 <= param1_idx < len(self.LINE_COLORS):
                    self.selected_line_index = param1_idx
                    reward += 0.01  # Mała nagroda za zmianę kontekstu
                else:
                    reward -= 0.01  # Kara za nieprawidłowy indeks koloru linii

        # --- 2. Ewolucja środowiska (niezależna od akcji agenta) ---
        # Spawnowanie pasażerów na stacjach
        for station in self.stations:
            timer_increase_value = random.randint(1, 20 + (round(self.score / 20)))
            station.passenger_timer += timer_increase_value
            if station.passenger_timer >= self.PASSENGER_SPAWN_RATE:
                station.passenger_timer = 0
                if not station.add_passenger(self.stations, self):
                    self.game_over = True
                    reward -= 50  # Duża kara za game over
                    terminated = True

        # Ruch pociągów
        trains_to_remove = []
        for train in self.trains:
            if not train.update(self):
                trains_to_remove.append(train)
            else:
                if len(train.passengers) > 0:
                    reward += 0.001  # Mała nagroda za każdy pasażera w ruchu (motywacja do dowożenia)

        for train in trains_to_remove:
            self.trains.remove(train)
            reward -= 0.5  # Kara za usunięcie pociągu (zazwyczaj dzieje się to z powodu błędu)

        self.all_passengers_plan_update()

        # Spawnowanie nowych stacji
        if not self.game_over:
            if self.spawned_stations_this_week_count < self.stations_to_spawn_this_week:
                if self.spawned_stations_this_week_count < len(self.STATION_SPAWN_TIMES) and \
                        self.week_timer >= self.STATION_SPAWN_TIMES[self.spawned_stations_this_week_count]:
                    spawned = False
                    attempts = 0
                    max_attempts = 100
                    while not spawned and attempts < max_attempts and len(self.stations) < self.MAX_STATIONS:
                        neighbour_station = random.choice(self.stations)
                        x = random.randint(max(20, round(neighbour_station.pos.x - self.SCREEN_WIDTH * 0.12)),
                                           min(self.SCREEN_WIDTH - 20,
                                               round(neighbour_station.pos.x + self.SCREEN_WIDTH * 0.12)))
                        y = random.randint(max(20, round(neighbour_station.pos.y - self.SCREEN_HEIGHT * 0.12)),
                                           min(self.SCREEN_HEIGHT - 20,
                                               round(neighbour_station.pos.y + self.SCREEN_HEIGHT * 0.12)))

                        shape = random.choice(self.STATION_SHAPES)
                        pos = pygame.Vector2(x, y)
                        if all(pos.distance_to(s.pos) > self.STATION_MIN_DISTANCE for s in self.stations):
                            new_station = Station(x, y, shape, env=self)
                            self.stations.append(new_station)
                            self.G.add_node(new_station.station_id, pos=(new_station.pos.x, new_station.pos.y))
                            self.all_passengers_plan_update()
                            spawned = True
                            self.spawned_stations_this_week_count += 1
                        attempts += 1

                    if not spawned and len(self.stations) < self.MAX_STATIONS:
                        pass
                    elif len(self.stations) >= self.MAX_STATIONS:
                        self.spawned_stations_this_week_count = self.stations_to_spawn_this_week

        # Upływ czasu tygodnia i koniec tygodnia
        self.week_timer += 1
        if self.week_timer >= self.WEEK_DURATION:
            self.week_timer = 0
            self.week_number += 1
            self.available_trains += 1
            self.spawned_stations_this_week_count = 0
            reward += 10  # Nagroda za przetrwanie tygodnia

        # --- 4. Sprawdzanie warunków zakończenia ---
        if self.game_over:
            terminated = True

        # Aktualizacja info
        info["week_number"] = self.week_number
        info["available_trains"] = self.available_trains
        info["score"] = self.score

        # Pobranie nowej obserwacji
        observation = self._get_obs()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            # Wypełnij ekran tłem
            self.screen.fill(self.BLACK)

            # Rysuj linie
            for line in self.lines:
                line.draw(self.screen)

            # Rysuj stacje
            for station in self.stations:
                station.draw(self.screen, self)  # Przekazujemy 'self' jako 'env'

            # Rysuj pociągi
            for train in self.trains:
                train.draw(self.screen)

            # Rysuj UI (wynik, tydzień, dostępne pociągi itp.)
            self.draw_ui(self.screen, self.font, self.font_small, self.score, self.week_number,
                         self.available_trains, self.deploy_train_mode, self.week_timer, self.WEEK_DURATION)

            # Jeśli gra się zakończyła, wyświetl ekran "Game Over"
            if self.game_over:
                overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 180))  # Półprzezroczysta czarna nakładka
                self.screen.blit(overlay, (0, 0))

                game_over_text_line1 = self.font_large.render("Game Over", True, self.RED)
                game_over_text_line2 = self.font.render(f"Final Score: {self.score}", True, self.WHITE)
                game_over_text_line3 = self.font.render(f"Reached Week: {self.week_number}", True, self.WHITE)

                rect1 = game_over_text_line1.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 40))
                rect2 = game_over_text_line2.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 + 20))
                rect3 = game_over_text_line3.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 + 50))

                self.screen.blit(game_over_text_line1, rect1)
                self.screen.blit(game_over_text_line2, rect2)
                self.screen.blit(game_over_text_line3, rect3)

            # Odśwież wyświetlacz Pygame
            pygame.display.flip()
            # Ogranicz klatki na sekundę
            self.clock.tick(self.FPS)

        return None

    def close(self):
        """Zamyka środowisko Pygame."""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None