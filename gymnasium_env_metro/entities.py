import pygame
from typing import List

from data_models import StationModel, LineModel, TrainModel
import config
from utils import draw_passenger_icon

class Station:
    def __init__(self, data : StationModel):
        self.data = data
        self.radius = config.STATION_RADIUS
        self.color = config.WHITE
        self.capacity = config.STATION_CAPACITY
        self.pos = pygame.Vector2(self.data.pos)

    def draw(self, screen : pygame.Surface):
        line_thickness = config.STATION_BORDER_THICKNESS + (2 if self.data.is_overcrowded else 0)
        outline_color = config.RED if self.data.is_overcrowded else self.color

        if self.data.shape == 'circle':
            pygame.draw.circle(screen, outline_color, self.data.pos, self.radius, line_thickness)
        elif self.data.shape == 'square':
            pygame.draw.rect(screen, outline_color, (self.data.pos[0] - self.radius, self.data.pos[1] - self.radius, self.radius*2, self.radius*2), line_thickness)
        elif self.data.shape == 'triangle':
            points = [
                (self.data.pos[0], self.data.pos[1] - self.radius),
                (self.data.pos[0]- self.radius, self.data.pos[1] + self.radius * 0.8),
                (self.data.pos[0] + self.radius, self.data.pos[1] + self.radius * 0.8)]
            pygame.draw.polygon(screen, outline_color, points, line_thickness)
        elif self.data.shape == 'star':
             points = [
                (self.data.pos[0], self.data.pos[1] - self.radius * 1.2),
                (self.data.pos[0] + self.radius * 0.35, self.data.pos[1] - self.radius * 0.35),
                (self.data.pos[0] + self.radius * 1.2, self.data.pos[1]),
                (self.data.pos[0] + self.radius * 0.35, self.data.pos[1] + self.radius * 0.35),
                (self.data.pos[0], self.data.pos[1] + self.radius * 1.2),
                (self.data.pos[0] - self.radius * 0.35, self.data.pos[1] + self.radius * 0.35),
                (self.data.pos[0] - self.radius * 1.2, self.data.pos[1]),
                (self.data.pos[0] - self.radius * 0.35, self.data.pos[1] - self.radius * 0.35)
            ]
             pygame.draw.polygon(screen, outline_color, points, line_thickness)
        elif self.data.shape == 'rombus':
            points = [
                (self.data.pos[0], self.data.pos[1] - self.radius),
                (self.data.pos[0] + self.radius, self.data.pos[1]),
                (self.data.pos[0], self.data.pos[1] + self.radius),
                (self.data.pos[0] - self.radius, self.data.pos[1]),
            ]
            pygame.draw.polygon(screen, outline_color, points, line_thickness+1)


        num_passengers = len(self.data.passengers)
        if num_passengers > 0:
            icon_size = config.PASSENGER_ICON_SIZE
            spacing = config.PASSENGER_ICON_SPACING
            total_width = num_passengers * icon_size + max(0, num_passengers - 1) * spacing

            row_y_center = self.data.pos[1] - self.radius - 15
            start_x_center = self.data.pos[0] - total_width / 2 + icon_size / 2

            for i, passenger in enumerate(self.data.passengers):
                icon_center_x = start_x_center + i * (icon_size + spacing)
                icon_center_pos = pygame.Vector2(icon_center_x, row_y_center)
                icon_color = config.WHITE

                draw_passenger_icon(screen, passenger.target_shape, icon_color, icon_center_pos, icon_size)
    def is_clicked(self, mouse_pos):
        return self.pos.distance_to(mouse_pos) <= self.radius + 5

class Line:
    def __init__(self, data: LineModel, all_stations: List[Station]):
            self.data = data
            self._update_station_objects(all_stations)

    def _update_station_objects(self, all_stations: List[Station]):
            """Synchronizuje listę obiektów stacji z listą ID w modelu danych."""
            station_map = {s.data.station_id: s for s in all_stations}
            self.stations = [station_map[sid] for sid in self.data.station_ids if sid in station_map]

    def add_station(self, station: Station, all_stations: List[Station], index: int = -1):
        station_id = station.data.station_id
        if station_id in self.data.station_ids: return False

        # POPRAWKA: Najpierw modyfikujemy dane
        if index == -1:
            self.data.station_ids.append(station_id)
        else:
            self.data.station_ids.insert(index, station_id)

        # POPRAWKA: Potem odświeżamy obiekty
        self._update_station_objects(all_stations)
        self.check_loop()
        return True

    def remove_station(self, station: Station, all_stations: List[Station]):
        station_id = station.data.station_id
        if station_id in self.data.station_ids:
            # POPRAWKA: Najpierw modyfikujemy dane
            self.data.station_ids = [sid for sid in self.data.station_ids if sid != station_id]

            # POPRAWKA: Potem odświeżamy obiekty
            self._update_station_objects(all_stations)
            self.check_loop()
            return True
        return False

    def check_loop(self):
        self.data.is_loop = len(self.data.station_ids) > 2 and self.data.station_ids[0] == self.data.station_ids[-1]

    def draw(self, screen: pygame.Surface):
        if len(self.stations) >= 2:
            points = [s.data.pos for s in self.stations]
            pygame.draw.lines(screen, self.data.color, self.data.is_loop, points, 5)

    def has_station(self, target_id):
        return target_id in self.data.station_ids


    """Nowe metody z Train->Line"""
    def get_station_by_id(self, station_id: str) -> Station | None:
        """Zwraca obiekt stacji na podstawie jej ID, jeśli należy do tej linii."""
        for station in self.stations:
            if station.data.station_id == station_id:
                return station
        return None

    def get_next_station_id(self, current_station_id: str, direction: int) -> str | None:
        """Oblicza ID następnej stacji na trasie."""
        try:
            current_index = self.data.station_ids.index(current_station_id)
        except ValueError:
            return None  # Stacja nie należy do tej linii

        num_stations = len(self.data.station_ids)
        if self.data.is_loop:
            # Logika dla pętli (zawsze idzie do przodu)
            next_index = (current_index + direction) % (num_stations - 1)  # -1 bo pętla ma stację startową na końcu
        else:
            # Logika dla linii prostej (może zmieniać kierunek na końcach)
            next_index = current_index + direction
            if not (0 <= next_index < num_stations):
                return None  # Pociąg dojechał do końca, potrzebuje zmiany kierunku

        return self.data.station_ids[next_index]


class Train:
    def __init__(self, data: TrainModel, line: Line):
        self.data = data
        self.line = line
        self.pos = pygame.Vector2(self.data.pos)

        self.speed = config.TRAIN_SPEED
        self.capacity = config.TRAIN_CAPACITY

        self.image = pygame.Surface((config.TRAIN_SIZE, config.TRAIN_SIZE * 2), pygame.SRCALPHA)
        self.image.fill(self.data.line_color)
        pygame.draw.rect(self.image, config.WHITE, (0, 0, config.TRAIN_SIZE, config.TRAIN_SIZE * 2), 1)

    def update(self, env):
        if not self.line or len(self.line.stations) < 2:
            return False

        target_station = self.line.get_station_by_id(self.data.target_station_id)
        if not target_station:
            return False

        target_pos = target_station.pos
        move_vector = target_pos - self.pos
        distance_to_target = move_vector.length()

        if distance_to_target < self.speed:
            # DOJAZD NA STACJĘ
            self.pos = target_pos
            self.data.pos = tuple(self.pos)
            self.data.current_station_id = self.data.target_station_id
            arrived_station = target_station

            # WYSADZANIE PASAŻERÓW
            passengers_staying = []
            for p_model in self.data.passengers:
                if p_model.target_shape == arrived_station.data.shape:
                    env.score += 1
                else:
                    passengers_staying.append(p_model)
            self.data.passengers = passengers_staying

            # ZABIERANIE PASAŻERÓW
            passengers_to_board = []
            passengers_staying_on_station = []

            for p_model in arrived_station.data.passengers:
                can_board = (
                        len(self.data.passengers) + len(passengers_to_board) < self.capacity and
                        p_model.travel_list and
                        self.line.has_station(p_model.travel_list[0])
                )
                if can_board:
                    passengers_to_board.append(p_model)
                else:
                    passengers_staying_on_station.append(p_model)

            # Potem aktualizujemy obie listy
            if passengers_to_board:
                self.data.passengers.extend(passengers_to_board)
                arrived_station.data.passengers = passengers_staying_on_station

            # WYZNACZANIE NOWEGO CELU
            next_id = self.line.get_next_station_id(self.data.current_station_id, self.data.direction)
            if next_id is None:
                self.data.direction *= -1
                next_id = self.line.get_next_station_id(self.data.current_station_id, self.data.direction)

            if next_id:
                self.data.target_station_id = next_id
            else:
                return False
        else:
            # RUCH MIĘDZY STACJAMI
            if distance_to_target > 0:
                move_vector.normalize_ip()
                self.pos += move_vector * self.speed
                self.data.pos = tuple(self.pos)

        return True

    def draw(self, screen: pygame.Surface):
        target_station = self.line.get_station_by_id(self.data.target_station_id)
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