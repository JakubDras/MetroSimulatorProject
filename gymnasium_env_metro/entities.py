import pygame
import random
from typing import List

from .data_models import StationModel, LineModel, TrainModel
from . import config
from .utils import draw_passenger_icon


class Station:
    def __init__(self, data : StationModel):
        self.data = data
        self.radius = config.STATION_RADIUS
        self.color = config.WHITE
        self.capacity = config.STATION_CAPACITY
        self.pos = pygame.Vector2(self.data.pos)

    def draw(self, screen):
        line_thickness = config.STATION_BORDER_THICKNESS + (2 if self.data.is_overcrowded else 0)
        outline_color = config.RED if self.data.is_overcrowded else self.color

        if self.data.shape == 'circle':
            pygame.draw.circle(screen, outline_color, self.pos, self.radius, line_thickness)
        elif self.data.shape == 'square':
            pygame.draw.rect(screen, outline_color, (self.pos.x - self.radius, self.pos.y - self.radius, self.radius*2, self.radius*2), line_thickness)
        elif self.data.shape == 'triangle':
            points = [
                (self.pos.x, self.pos.y - self.radius),
                (self.pos.x - self.radius, self.pos.y + self.radius * 0.8),
                (self.pos.x + self.radius, self.pos.y + self.radius * 0.8)]
            pygame.draw.polygon(screen, outline_color, points, line_thickness)
        elif self.data.shape == 'star':
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
        elif self.data.shape == 'rombus':
            points = [
                (self.pos.x, self.pos.y - self.radius),
                (self.pos.x + self.radius, self.pos.y),
                (self.pos.x, self.pos.y + self.radius),
                (self.pos.x - self.radius, self.pos.y),
            ]
            pygame.draw.polygon(screen, outline_color, points, line_thickness+1)


        num_passengers = len(self.data.passengers)
        if num_passengers > 0:
            icon_size = config.PASSENGER_ICON_SIZE
            spacing = config.PASSENGER_ICON_SPACING
            total_width = num_passengers * icon_size + max(0, num_passengers - 1) * spacing

            row_y_center = self.pos.y - self.radius - 15
            start_x_center = self.pos.x - total_width / 2 + icon_size / 2

            for i, passenger in enumerate(self.data.passengers):
                icon_center_x = start_x_center + i * (icon_size + spacing)
                icon_center_pos = pygame.Vector2(icon_center_x, row_y_center)
                icon_color = config.WHITE

                draw_passenger_icon(screen, passenger.target_shape, icon_color, icon_center_pos, icon_size)

    def is_clicked(self, mouse_pos):
        return self.pos.distance_to(mouse_pos) <= self.radius + 5
class Line:
    def __init__(self, data: LineModel):
        self.data = data
        self.stations : List[Station] = []
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
        self.data.is_loop = len(self.stations) > 2 and self.stations[0] == self.stations[-1]

    def draw(self, screen):
        if len(self.stations) >= 2:
            points = [s.pos for s in self.stations]
            pygame.draw.lines(screen, self.data.color, self.data.is_loop, points, 5)

    def has_station(self, target_id):
        return any(station.data.station_id == target_id for station in self.stations)
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

    def find_station_index_by_id(self, station_id):
        for i, station in enumerate(self.line.stations):
            if station.data.station_id == station_id:
                return i
        return -1

    def get_station_by_id(self, station_id):
        for station in self.line.stations:
            if station.data.station_id == station_id:
                return station
        return None

    def update(self, env):
        passengers_delivered_count = 0

        if not self.line or len(self.line.stations) < 2:
            return False, 0

        current_index = self.find_station_index_by_id(self.data.current_station_id)
        target_index = self.find_station_index_by_id(self.data.target_station_id)

        if current_index == -1 or target_index == -1:
            print("Błąd: stacja aktualna lub docelowa nie istnieje na linii. Usuwam pociąg.")
            return False, 0

        target_station = self.line.stations[target_index]
        target_pos = target_station.pos
        move_vector = target_pos - self.pos
        distance_to_target = move_vector.length()

        if distance_to_target < self.speed:
            self.pos = pygame.Vector2(target_pos)
            self.data.current_station_id = self.data.target_station_id
            arrived_station = self.get_station_by_id(self.data.current_station_id)

            num_stations = len(self.line.stations)
            if num_stations < 2:
                return False, 0

            current_index = self.find_station_index_by_id(self.data.current_station_id)

            if self.line.data.is_loop:
                next_index = (current_index + self.data.direction) % num_stations
                if self.data.direction == -1 and next_index == num_stations - 1:
                    next_index -= 1
                self.data.target_station_id = self.line.stations[next_index].data.station_id
            else:
                next_index = current_index + self.data.direction
                if 0 <= next_index < num_stations:
                    self.data.target_station_id = self.line.stations[next_index].data.station_id
                else:
                    self.data.direction *= -1
                    next_index = current_index + self.data.direction
                    if 0 <= next_index < num_stations:
                        self.data.target_station_id = self.line.stations[next_index].data.station_id
                    else:
                        print("Nie udało się znaleźć nowej stacji, resetuję pociąg.")
                        self.data.current_station_id = self.line.stations[0].data.station_id
                        self.data.target_station_id = self.line.stations[1 % num_stations].data.station_id
                        self.data.direction = 1
                        self.pos = pygame.Vector2(self.line.stations[0].pos)
                        return False, 0

            passengers_staying = []
            for p in self.data.passengers:
                if p.target_shape == arrived_station.data.shape:
                    env.score += 1
                    passengers_delivered_count += 1
                elif len(p.travel_list) == 0:
                    arrived_station.data.passengers.append(p)
                    # env.travel_planner_for_new_passager(p)
                    env.travel_planner_for_new_passenger(p)
                elif p.travel_list[0] == arrived_station.data.station_id:
                    p.travel_list.pop(0)
                    arrived_station.data.passengers.append(p)
                else:
                    passengers_staying.append(p)
            self.data.passengers = passengers_staying

            if not arrived_station.data.is_overcrowded:
                boarding_list = list(arrived_station.data.passengers)
                random.shuffle(boarding_list)

                for p in boarding_list:
                    if p.travel_list and len(self.data.passengers) < self.capacity:

                        should_board = False

                        if self.line.data.is_loop:
                            if self.line.has_station(p.travel_list[0]):
                                should_board = True
                        else:
                            if self.line.has_station(p.travel_list[0]) and p.travel_list[
                                0] == self.data.target_station_id:
                                should_board = True

                        if should_board:
                            self.data.passengers.append(p)
                            if p in arrived_station.data.passengers:
                                arrived_station.data.passengers.remove(p)

            if arrived_station.data.is_overcrowded and len(arrived_station.data.passengers) < arrived_station.capacity:
                arrived_station.data.is_overcrowded = False

        else:
            if distance_to_target > 0:
                try:
                    move_vector.normalize_ip()
                    self.pos += move_vector * self.speed
                except ValueError:
                    self.pos = pygame.Vector2(target_pos)
            else:
                self.pos = pygame.Vector2(target_pos)

        return True, passengers_delivered_count

    def draw(self, screen):
        if len(self.line.stations) < 2:
            return
        target_station = self.get_station_by_id(self.data.target_station_id)
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