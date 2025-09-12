import math
import networkx as nx
import pygame
import sys
import random

SCREEN_WIDTH = 900
SCREEN_HEIGHT = 600
FPS = 60

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PINK = (255,192,203)
GRAY = (100, 100, 100)
GREEN_BLUE = (50,166,154)
LIGHT_GRAY = (200, 200, 200)
LINE_COLORS = [RED, GREEN, BLUE, YELLOW, PINK, GREEN_BLUE]

STATION_RADIUS = 9
STATION_BORDER_THICKNESS = 3
STATION_CAPACITY = 6
STATION_MIN_DISTANCE = 70
STATION_SHAPES = ['circle', 'square', 'triangle', 'star']
MAX_STATIONS = 40
PASSENGER_SPAWN_RATE = 70

PASSENGER_ICON_SIZE = 10

PASSENGER_ICON_SPACING = 5

TRAIN_SPEED = 1.2
TRAIN_CAPACITY = 6
TRAIN_SIZE = 10
TRAIN_ID_NUMBER = 0
STATION_ID = 0

# --- Game State ---
score = 0
week_number = 1
week_timer = 0
WEEK_DURATION = FPS * 60

available_trains = 3
deploy_train_mode = False
TRAIN_BUTTON_RECT = pygame.Rect(SCREEN_WIDTH - 120, 70, 100, 40)

# --- NEW STATIONS ---
stations_to_spawn_this_week = 5
STATION_SPAWN_TIMES = [((WEEK_DURATION // 5)-1)*1,((WEEK_DURATION // 5)-1)*2,((WEEK_DURATION // 5)-1)*3,((WEEK_DURATION // 5)-1)*4,((WEEK_DURATION // 5)-1)*5]
spawned_stations_this_week_count = 0

# --- WEEK TIMER DISPLAY ---
WEEK_BAR_RECT = pygame.Rect(20, 60, 150, 15)
WEEK_BAR_COLOR = YELLOW
WEEK_BAR_BG_COLOR = GRAY


# --- Inicjalizacja ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("MiniMetro Clone")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 28)
font_small = pygame.font.Font(None, 20)
font_large = pygame.font.Font(None, 72)

def get_train_id():
    global TRAIN_ID_NUMBER
    TRAIN_ID_NUMBER += 1
    name = f"T{TRAIN_ID_NUMBER}"
    return name

def get_station_id():
    global STATION_ID
    STATION_ID += 1
    name = f"S{STATION_ID}"
    return name

def draw_passenger_icon(surface, shape, color, center_pos, size):
    half_size = size // 2
    outline_thickness = 1

    if shape == 'circle':
        pygame.draw.circle(surface, color, center_pos, half_size)
        pygame.draw.circle(surface, BLACK, center_pos, half_size, outline_thickness)
    elif shape == 'square':
        rect = pygame.Rect(center_pos.x - half_size, center_pos.y - half_size, size, size)
        pygame.draw.rect(surface, color, rect)
        pygame.draw.rect(surface, BLACK, rect, outline_thickness)
    elif shape == 'triangle':
        points = [
            (center_pos.x, center_pos.y - half_size),
            (center_pos.x - half_size, center_pos.y + half_size * 0.8),
            (center_pos.x + half_size, center_pos.y + half_size * 0.8)]
        pygame.draw.polygon(surface, color, points)
        pygame.draw.polygon(surface, BLACK, points, outline_thickness)
    elif shape == 'star':
        points = [
            (center_pos.x, center_pos.y - size * 0.6),
            (center_pos.x + size * 0.15, center_pos.y - size * 0.15),
            (center_pos.x + size * 0.6, center_pos.y),
            (center_pos.x + size * 0.15, center_pos.y + size * 0.15),
            (center_pos.x, center_pos.y + size * 0.6),
            (center_pos.x - size * 0.15, center_pos.y + size * 0.15),
            (center_pos.x - size * 0.6, center_pos.y),
            (center_pos.x - size * 0.15, center_pos.y - size * 0.15)
        ]
        pygame.draw.polygon(surface, color, points)
        pygame.draw.polygon(surface, BLACK, points, outline_thickness)

class Passenger:
    def __init__(self, origin_station, target_shape):
        self.origin_station = origin_station
        self.target_shape = target_shape
        self.travel_list = []

class Station:
    def __init__(self, x, y, shape):
        self.station_id = get_station_id()
        self.pos = pygame.Vector2(x, y)
        self.shape = shape
        self.radius = STATION_RADIUS
        self.color = WHITE
        self.passengers = []
        self.capacity = STATION_CAPACITY
        self.is_overcrowded = False

    def draw(self, screen):
        line_thickness = STATION_BORDER_THICKNESS + (2 if self.is_overcrowded else 0)
        outline_color = RED if self.is_overcrowded else self.color

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

        num_passengers = len(self.passengers)
        if num_passengers > 0:
            icon_size = PASSENGER_ICON_SIZE
            spacing = PASSENGER_ICON_SPACING
            total_width = num_passengers * icon_size + max(0, num_passengers - 1) * spacing

            row_y_center = self.pos.y - self.radius - 15
            start_x_center = self.pos.x - total_width / 2 + icon_size / 2

            for i, passenger in enumerate(self.passengers):
                icon_center_x = start_x_center + i * (icon_size + spacing)
                icon_center_pos = pygame.Vector2(icon_center_x, row_y_center)
                icon_color = WHITE

                draw_passenger_icon(screen, passenger.target_shape, icon_color, icon_center_pos, icon_size)


    def add_passenger(self, all_stations):
        if len(self.passengers) >= self.capacity:
            self.is_overcrowded = True
            return False

        possible_targets = [s.shape for s in all_stations if s.shape != self.shape]
        if not possible_targets:
             if len(all_stations) > 1:
                 target_shape = random.choice([s for s in STATION_SHAPES if s != self.shape])
                 p = Passenger(self,target_shape)

                 travel_planner_for_new_passager(p,stations,G)

                 self.passengers.append(p)
                 return True
             else:
                return True

        target_shape = random.choice(possible_targets)
        p = Passenger(self, target_shape)
        travel_planner_for_new_passager(p,stations,G)
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
    def __init__(self, line):
        if not line or len(line.stations) < 2:
            raise ValueError("Pociąg musi być przypisany do linii z co najmniej dwiema stacjami.")

        self.id = get_train_id()
        self.line = line
        self.color = line.color
        self.current_station_index = 0
        self.target_station_index = 1 % len(line.stations)
        self.pos = pygame.Vector2(self.line.stations[0].pos)
        self.direction = 1
        self.speed = TRAIN_SPEED
        self.size = TRAIN_SIZE
        self.passengers = []
        self.capacity = TRAIN_CAPACITY

        self.image = pygame.Surface((self.size, self.size * 2), pygame.SRCALPHA)
        self.image.fill(self.color)
        pygame.draw.rect(self.image, WHITE, (0, 0, self.size, self.size * 2), 1)


    def update(self):
        if not self.line or len(self.line.stations) < 2:
             return False

        num_stations = len(self.line.stations)

        if not (0 <= self.target_station_index < num_stations):
             print(f"Błąd indeksu celu pociągu: target={self.target_station_index}, len={num_stations}. Próbuję naprawić.")
             next_index_attempt = self.current_station_index + self.direction
             if 0 <= next_index_attempt < num_stations:
                  self.target_station_index = next_index_attempt
             else:
                  self.current_station_index = 0
                  self.target_station_index = 1 % num_stations if num_stations > 0 else 0
                  self.direction = 1
                  self.pos = pygame.Vector2(self.line.stations[0].pos)
                  if not (0 <= self.target_station_index < num_stations):
                       print("Nie udało się naprawić indeksu pociągu, usuwam pociąg.")
                       return False

        target_station = self.line.stations[self.target_station_index]
        target_pos = target_station.pos

        move_vector = target_pos - self.pos
        distance_to_target = move_vector.length()

        if distance_to_target < self.speed:
            self.pos = pygame.Vector2(target_pos)
            self.current_station_index = self.target_station_index
            arrived_station = self.line.stations[self.current_station_index]

            passengers_staying = []
            for p in self.passengers:
                if p.target_shape == arrived_station.shape:
                    global score
                    score += 1
                elif len(p.travel_list) == 0:
                    arrived_station.passengers.append(p)
                    travel_planner_for_new_passager(p,stations,G)
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
                        if len(self.passengers) < self.capacity and self.line.has_station(p.travel_list[0]):
                            self.passengers.append(p)
                            if p in arrived_station.passengers:
                                arrived_station.passengers.remove(p)

            if arrived_station.is_overcrowded and len(arrived_station.passengers) < arrived_station.capacity:
                 arrived_station.is_overcrowded = False

            num_stations = len(self.line.stations)
            if num_stations < 2:
                 return False

            if self.line.is_loop:
                 next_index_candidate = self.current_station_index + self.direction
                 if next_index_candidate >= num_stations:
                     self.target_station_index = 0
                 elif next_index_candidate < 0:
                      self.target_station_index = num_stations -1
                 else:
                      self.target_station_index = next_index_candidate

            else:
                next_index = self.current_station_index + self.direction
                if 0 <= next_index < num_stations:
                    self.target_station_index = next_index
                else:
                    self.direction *= -1
                    self.target_station_index = self.current_station_index + self.direction
                    if not (0 <= self.target_station_index < num_stations):
                         print("Naprawianie celu pociągu po zmianie kierunku na końcu linii.")
                         self.current_station_index = 0
                         self.target_station_index = 1 % num_stations if num_stations > 0 else 0
                         self.direction = 1
                         self.pos = pygame.Vector2(self.line.stations[0].pos)
                         if not (0 <= self.target_station_index < num_stations):
                             return False
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
        if len(self.line.stations) < 2: return

        target_station = self.line.stations[self.target_station_index]
        move_direction = target_station.pos - self.pos

        if move_direction.length_squared() > 0.1:
             angle = pygame.math.Vector2(0, -1).angle_to(move_direction)
             rotated_image = pygame.transform.rotate(self.image, angle)
             rect = rotated_image.get_rect(center=self.pos)
             screen.blit(rotated_image, rect)
        else:
             rect = self.image.get_rect(center=self.pos)
             screen.blit(self.image, rect)

def travel_planner_for_new_passager(new_passenger, stations, g):
    final_destination_station = None
    final_shortest_distance_in_g = float('inf')
    found_path_for_passenger = False
    try:
        target_stations = [t for t in stations if t.shape == new_passenger.target_shape]

        for station in target_stations:
            try:
                distance = nx.shortest_path_length(g, source=new_passenger.origin_station.station_id, target=station.station_id, weight='weight')
                if distance < final_shortest_distance_in_g:
                    final_destination_station = station
                    final_shortest_distance_in_g = distance
                    found_path_for_passenger = True
            except nx.NetworkXNoPath:
                pass

        if found_path_for_passenger and final_destination_station is not None:
            path = nx.dijkstra_path(g, source=new_passenger.origin_station.station_id, target=final_destination_station.station_id, weight='weight')
            new_passenger.travel_list = path[1:]
        else:
            new_passenger.travel_list = []

    except Exception as e:
        pass

def all_passengers_plan_update(stations, g):
    for s in stations:
        for p in s.passengers:
            final_destination_station = None
            final_shortest_distance_in_g = float('inf')
            found_path_for_passenger = False
            try:
                target_stations = [t for t in stations if t.shape == p.target_shape]
                for station in target_stations:
                    try:
                        distance = nx.shortest_path_length(g, source=s.station_id, target=station.station_id, weight='weight')
                        if distance < final_shortest_distance_in_g:
                            final_destination_station = station
                            final_shortest_distance_in_g = distance
                            found_path_for_passenger = True
                    except nx.NetworkXNoPath:
                        pass
                if found_path_for_passenger and final_destination_station is not None:
                    path = nx.dijkstra_path(g, source=s.station_id, target=final_destination_station.station_id, weight='weight')
                    p.travel_list = path[1:]
                else:
                    p.travel_list = []
            except Exception as e:
                print(f"Nieoczekiwany błąd podczas AllPassengersPlanUpdate dla pasażera na {s.station_id}: {e}")
                p.travel_list = []

def euclidean_distance(start_station, end_station):
    pos1 = (start_station.pos.x,start_station.pos.y)
    pos2 = (end_station.pos.x,end_station.pos.y)
    return round(math.dist(pos1,pos2))

def calculate_edge_weight(graph, node1_id, node2_id):
    pos1 = graph.nodes[node1_id]['pos']
    pos2 = graph.nodes[node2_id]['pos']
    return round(math.dist(pos1,pos2))

def get_station_at(pos, stations):
    for station in stations:
        if station.is_clicked(pos):
            return station
    return None

ui_circles_rects = []

def draw_ui_circles(screen, lines, selected_line_index):
    global ui_circles_rects
    ui_circles_rects = []
    start_x = 20
    y = 30
    radius = 15
    spacing = 50
    for i in range(len(LINE_COLORS)):
        color = LINE_COLORS[i]
        line_exists = any(line.color == color for line in lines)
        rect = pygame.Rect(start_x + i * spacing - radius, y - radius, radius * 2, radius * 2)
        ui_circles_rects.append(rect)
        pygame.draw.circle(screen, color, rect.center, radius)
        if i == selected_line_index:
             pygame.draw.circle(screen, WHITE, rect.center, radius + 2, 3)
        elif line_exists:
             pygame.draw.circle(screen, WHITE, rect.center, radius, 1)
        else:
             pygame.draw.circle(screen, GRAY, rect.center, radius, 1)

def draw_ui(screen, font, font_small, score, week_number, available_trains, deploy_train_mode, week_timer, week_duration):
    draw_ui_circles(screen, lines, selected_line_index)

    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (SCREEN_WIDTH - 180, 10))
    week_text = font.render(f"Week: {week_number}", True, WHITE)
    screen.blit(week_text, (SCREEN_WIDTH - 180, 40))

    pygame.draw.rect(screen, WEEK_BAR_BG_COLOR, WEEK_BAR_RECT)
    progress = week_timer / week_duration
    current_bar_width = int(WEEK_BAR_RECT.width * progress)
    pygame.draw.rect(screen, WEEK_BAR_COLOR, (WEEK_BAR_RECT.left, WEEK_BAR_RECT.top, current_bar_width, WEEK_BAR_RECT.height))
    pygame.draw.rect(screen, WHITE, WEEK_BAR_RECT, 1) # Ramka paska

    button_color = GREEN if deploy_train_mode else BLUE
    pygame.draw.rect(screen, button_color, TRAIN_BUTTON_RECT)
    pygame.draw.rect(screen, WHITE, TRAIN_BUTTON_RECT, 3)
    train_button_text = font_small.render(f"Train ({available_trains})", True, WHITE)
    text_rect = train_button_text.get_rect(center=TRAIN_BUTTON_RECT.center)
    screen.blit(train_button_text, text_rect)

# --- Main loop of game ---
stations = []
lines = []
trains = []
passenger_timer = 0
game_over = False
G = nx.MultiGraph()

# --- GAME START: Generating Initial Stations ---
initial_stations_count = 3
available_shapes = list(STATION_SHAPES)
random.shuffle(available_shapes)
initial_shapes = [available_shapes.pop() for _ in range(min(initial_stations_count, len(available_shapes)))]


x = random.randint(round(SCREEN_WIDTH* 0.25), round(SCREEN_WIDTH* 0.75))
y = random.randint(round(SCREEN_HEIGHT* 0.25), round(SCREEN_HEIGHT* 0.75))
shape = initial_shapes.pop() if initial_shapes else random.choice(STATION_SHAPES)
pos = pygame.Vector2(x, y)
if not stations or all(pos.distance_to(s.pos) > STATION_MIN_DISTANCE for s in stations):
    stations.append(Station(x, y, shape))
    print(f"Dodano początkową stację: {shape} w ({x}, {y})")

while len(stations) < initial_stations_count:
    stations_all = [s for s in stations]
    neighbour_station = random.choice(stations_all)
    x = random.randint(round(neighbour_station.pos.x - SCREEN_WIDTH * 0.15),
                       round(neighbour_station.pos.x + SCREEN_WIDTH * 0.15))
    y = random.randint(round(neighbour_station.pos.y - SCREEN_HEIGHT * 0.15),
                       round(neighbour_station.pos.y + SCREEN_HEIGHT * 0.15))
    while 50 > x or x > SCREEN_WIDTH - 50 or 50 > y or y > SCREEN_HEIGHT - 50:
        x = random.randint(round(neighbour_station.pos.x - SCREEN_WIDTH * 0.15),
                           round(neighbour_station.pos.x + SCREEN_WIDTH * 0.15))
        y = random.randint(round(neighbour_station.pos.y - SCREEN_HEIGHT * 0.15),
                           round(neighbour_station.pos.y + SCREEN_HEIGHT * 0.15))
    shape = initial_shapes.pop() if initial_shapes else random.choice(STATION_SHAPES)
    pos = pygame.Vector2(x, y)
    if not stations or all(pos.distance_to(s.pos) > STATION_MIN_DISTANCE for s in stations):
        stations.append(Station(x, y, shape))
        print(f"Dodano początkową stację: {shape} w ({x}, {y})")
for s in stations:
    G.add_node(s.station_id, pos=(s.pos.x,s.pos.y))

dragging = False
drag_start_station = None
drag_line_color = None
selected_line_index = 0

running = True
while running:
    mouse_pos = pygame.mouse.get_pos()

    # --- Drawing at the beginning of the loop ---
    screen.fill(BLACK)
    for line in lines: line.draw(screen)
    if dragging and drag_start_station: pygame.draw.line(screen, drag_line_color, drag_start_station.pos, mouse_pos, 3)
    for station in stations: station.draw(screen)
    trains = [train for train in trains if train.update()]
    for train in trains: train.draw(screen)

    # --- Drawing UI ---
    draw_ui(screen, font, font_small, score, week_number, available_trains, deploy_train_mode, week_timer, WEEK_DURATION)

    # --- Event handling ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            ui_action_taken = False

            # --- Right mouse button (Delete line/station) ---
            if event.button == 3:
                if game_over: continue

                for i, rect in enumerate(ui_circles_rects):
                    if rect.collidepoint(mouse_pos):
                        color_to_delete = LINE_COLORS[i]
                        line_to_delete = None
                        for line in lines:
                            if line.color == color_to_delete:
                                line_to_delete = line
                                break
                        if line_to_delete:
                            count = 0
                            for train in trains:
                                if train.color == line_to_delete.color:
                                    count += 1
                            trains = [t for t in trains if t.line != line_to_delete]
                            lines.remove(line_to_delete)
                            available_trains += count
                            print(f"Usunięto linię {color_to_delete}")

                            edges_to_remove = []
                            for u, v, key, data in G.edges(data=True, keys=True):
                                if data.get('color') == color_to_delete:
                                    edges_to_remove.append((u, v, key))
                            for u, v, key in edges_to_remove:
                                G.remove_edge(u, v, key)

                            all_passengers_plan_update(stations, G)

                            ui_action_taken = True
                            if i == selected_line_index:
                                 selected_line_index = 0
                                 if lines:
                                     drag_line_color = lines[0].color
                                 else:
                                     drag_line_color = LINE_COLORS[0]
                        break

            # --- Left mouse button ---
            elif event.button == 1:
                if game_over: continue

                if TRAIN_BUTTON_RECT.collidepoint(mouse_pos):
                    if available_trains > 0:
                        deploy_train_mode = not deploy_train_mode
                        dragging = False
                        drag_start_station = None
                    else:
                        print("Brak dostępnych pociągów.")
                    ui_action_taken = True

                if not ui_action_taken:
                    for i, rect in enumerate(ui_circles_rects):
                        if rect.collidepoint(mouse_pos):
                            selected_line_index = i
                            drag_line_color = LINE_COLORS[selected_line_index]
                            deploy_train_mode = False
                            ui_action_taken = True
                            break

                if not ui_action_taken and deploy_train_mode:
                    clicked_station = get_station_at(mouse_pos, stations)
                    if clicked_station:
                        line_to_add_train = None
                        for line in lines:
                             if line.color == LINE_COLORS[selected_line_index] and clicked_station in line.stations:
                                  line_to_add_train = line
                                  break
                        if line_to_add_train is None:
                            for line in lines:
                                 if clicked_station in line.stations:
                                      line_to_add_train = line
                                      break

                        if line_to_add_train and len(line_to_add_train.stations) >= 2:
                             trains.append(Train(line_to_add_train))
                             available_trains -= 1
                             deploy_train_mode = False
                             print(f"Rozmieszczono pociąg na linii {line_to_add_train.color}.")
                        elif line_to_add_train and len(line_to_add_train.stations) < 2:
                             print("Linia jest za krótka (wymaga >= 2 stacji), aby umieścić pociąg.")
                        else:
                            print("Kliknięta stacja nie jest na żadnej linii z co najmniej 2 stacjami.")

                elif not ui_action_taken and not deploy_train_mode:
                    station = get_station_at(mouse_pos, stations)
                    if station:
                        dragging = True
                        drag_start_station = station
                        drag_line_color = LINE_COLORS[selected_line_index]

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and dragging and drag_start_station and not game_over and not deploy_train_mode:
                end_station = get_station_at(mouse_pos, stations)
                if end_station:
                    line_to_modify = next((line for line in lines if line.color == drag_line_color), None)

                    if line_to_modify:
                        can_extend_end = line_to_modify.stations[-1] == drag_start_station
                        can_extend_start = line_to_modify.stations[0] == drag_start_station
                        is_closing_loop_end_to_start = can_extend_end and end_station == line_to_modify.stations[0]
                        is_closing_loop_start_to_end = can_extend_start and end_station == line_to_modify.stations[-1]
                        is_already_on_line = end_station in line_to_modify.stations

                        if len(line_to_modify.stations) >= 2 and (is_closing_loop_end_to_start or is_closing_loop_start_to_end):
                             unique_stations_count = len(set(line_to_modify.stations + [end_station]))
                             if unique_stations_count >= 3 or (unique_stations_count == 2 and line_to_modify.stations[0] == line_to_modify.stations[-1]):
                                if is_closing_loop_end_to_start:
                                    line_to_modify.add_station(end_station, index=-1)
                                else:
                                    line_to_modify.add_station(end_station, index=0)
                                if line_to_modify.is_loop:
                                   print(f"Zamknięto pętlę dla linii {line_to_modify.color}")
                                   G.add_edge(drag_start_station.station_id, end_station.station_id,
                                              key=str(line_to_modify.color),
                                              weight=euclidean_distance(drag_start_station, end_station),
                                              color=line_to_modify.color)
                                   all_passengers_plan_update(stations, G)
                                else:
                                   print(f"Stacja {end_station.shape} dodana do linii {line_to_modify.color}, ale nie utworzono pętli.")
                             else:
                                  print(f"Nie można zamknąć pętli, potrzeba co najmniej 3 unikalnych stacji.")

                        elif not is_already_on_line:
                            if can_extend_end:
                                line_to_modify.add_station(end_station, index=-1)
                                print(f"Przedłużono koniec linii {line_to_modify.color} o stację {end_station.shape}")
                                G.add_edge(drag_start_station.station_id, end_station.station_id,
                                           key=str(line_to_modify.color),
                                           weight=euclidean_distance(drag_start_station, end_station),
                                           color=line_to_modify.color)
                                all_passengers_plan_update(stations, G)

                            elif can_extend_start:
                                line_to_modify.add_station(end_station, index=0)
                                print(f"Przedłużono początek linii {line_to_modify.color} o stację {end_station.shape}")
                                G.add_edge(drag_start_station.station_id, end_station.station_id,
                                           key=str(line_to_modify.color),
                                           weight=euclidean_distance(drag_start_station, end_station),
                                           color=line_to_modify.color)
                                all_passengers_plan_update(stations, G)

                    else:
                        if drag_start_station != end_station:
                            if drag_line_color:
                                new_line = Line(drag_line_color)
                                new_line.add_station(drag_start_station)
                                new_line.add_station(end_station)
                                lines.append(new_line)
                                print(f"Utworzono nową linię {new_line.color} między {drag_start_station.shape} a {end_station.shape}")
                                G.add_edge(drag_start_station.station_id, end_station.station_id,
                                           key=str(new_line.color),
                                           weight=euclidean_distance(drag_start_station, end_station),
                                           color=new_line.color)
                                all_passengers_plan_update(stations, G)
                            else:
                                print("Nie wybrano koloru linii.")

            dragging = False
            drag_start_station = None

    # --- Game logic (outside of event handling) ---
    if not game_over:
        # pojawianie sie pasażerów
        passenger_timer += 0.4
        if passenger_timer >= PASSENGER_SPAWN_RATE:
            passenger_timer = 0
            stations_all = [s for s in stations]
            if stations_all:
                random_station = random.choice(stations_all)
                if not random_station.add_passenger(stations):
                    game_over = True
                    print(f"Game Over! Stacja {random_station.shape} ({random_station.pos}) przepełniona.")

        week_timer += 1

        if spawned_stations_this_week_count < stations_to_spawn_this_week:
            if spawned_stations_this_week_count < len(STATION_SPAWN_TIMES) and week_timer >= STATION_SPAWN_TIMES[spawned_stations_this_week_count]:
                spawned = False
                attempts = 0
                max_attempts = 100
                while not spawned and attempts < max_attempts and len(stations) < MAX_STATIONS:
                    stations_all = [s for s in stations]
                    neighbour_station = random.choice(stations_all)
                    x = random.randint(round(neighbour_station.pos.x - SCREEN_WIDTH*0.12), round(neighbour_station.pos.x + SCREEN_WIDTH*0.12))
                    y = random.randint(round(neighbour_station.pos.y - SCREEN_HEIGHT*0.12), round(neighbour_station.pos.y + SCREEN_HEIGHT*0.12))

                    while 20 > x or x > SCREEN_WIDTH - 20 or 20 > y or y > SCREEN_HEIGHT - 20:
                        x = random.randint(round(neighbour_station.pos.x - SCREEN_WIDTH * 0.12),
                                           round(neighbour_station.pos.x + SCREEN_WIDTH * 0.12))
                        y = random.randint(round(neighbour_station.pos.y - SCREEN_HEIGHT * 0.12),
                                           round(neighbour_station.pos.y + SCREEN_HEIGHT * 0.12))
                    shape = random.choice(STATION_SHAPES)
                    pos = pygame.Vector2(x, y)
                    if all(pos.distance_to(s.pos) > STATION_MIN_DISTANCE for s in stations):
                        s = Station(x, y, shape)
                        stations.append(s)
                        G.add_node(s.station_id,pos=(s.pos.x,s.pos.y))

                        print(f"Nowa stacja ({shape}) pojawiła się w Tygodniu {week_number}.")
                        spawned = True
                        spawned_stations_this_week_count += 1
                    attempts += 1

                if not spawned and len(stations) < MAX_STATIONS:
                     print(f"Nie udało się dodać nowej stacji w Tygodniu {week_number} po wielu próbach.")
                elif len(stations) >= MAX_STATIONS:
                     print("Osiągnięto maksymalną liczbę stacji.")
                     spawned_stations_this_week_count = stations_to_spawn_this_week

        if week_timer >= WEEK_DURATION:
            week_timer = 0
            week_number += 1
            available_trains += 1
            spawned_stations_this_week_count = 0
            print(f"Koniec Tygodnia {week_number-1}. Dostępny nowy pociąg! Rozpoczyna się Tydzień {week_number}.")

    if game_over:
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        screen.blit(overlay, (0, 0))

        game_over_text_line1 = font_large.render("Game Over", True, RED)
        game_over_text_line2 = font.render(f"Final Score: {score}", True, WHITE)
        game_over_text_line3 = font.render(f"Reached Week: {week_number}", True, WHITE)

        rect1 = game_over_text_line1.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 40))
        rect2 = game_over_text_line2.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 20))
        rect3 = game_over_text_line3.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50))

        screen.blit(game_over_text_line1, rect1)
        screen.blit(game_over_text_line2, rect2)
        screen.blit(game_over_text_line3, rect3)


    # --- Screen update ---
    pygame.display.flip()
    clock.tick(FPS)

# --- Game over ---
pygame.quit()
sys.exit()