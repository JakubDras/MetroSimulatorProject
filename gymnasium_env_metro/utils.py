import pygame
from .data_models import Shape, Color

def draw_passenger_icon(surface: pygame.Surface, shape: Shape, color: Color, center_pos: pygame.Vector2, size: int):
    half_size = size // 2
    outline_thickness = 1
    black = (0, 0, 0)

    if shape == 'circle':
        pygame.draw.circle(surface, color, center_pos, half_size)
        pygame.draw.circle(surface, black, center_pos, half_size, outline_thickness)
    elif shape == 'square':
        rect = pygame.Rect(center_pos.x - half_size, center_pos.y - half_size, size, size)
        pygame.draw.rect(surface, color, rect)
        pygame.draw.rect(surface, black, rect, outline_thickness)
    elif shape == 'triangle':
        points = [
            (center_pos.x, center_pos.y - half_size),
            (center_pos.x - half_size, center_pos.y + half_size * 0.8),
            (center_pos.x + half_size, center_pos.y + half_size * 0.8)
        ]
        pygame.draw.polygon(surface, color, points)
        pygame.draw.polygon(surface, black, points, outline_thickness)
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
        pygame.draw.polygon(surface, black, points, outline_thickness)
    elif shape == 'rombus':
        points = [
            (center_pos.x, center_pos.y - size * 0.5),
            (center_pos.x + size * 0.5, center_pos.y),
            (center_pos.x, center_pos.y + size * 0.5),
            (center_pos.x - size * 0.5, center_pos.y),
        ]
        pygame.draw.polygon(surface, color, points)
        pygame.draw.polygon(surface, black, points, outline_thickness)