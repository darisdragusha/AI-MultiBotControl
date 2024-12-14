import pygame
from src.core.constants import BLUE, GRAY, BLACK

class Button:
    def __init__(self, x, y, width, height, text, icon=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.icon = icon
        self.selected = False

    def draw(self, screen):
        color = BLUE if self.selected else GRAY
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 2)
        
        if self.text:
            font = pygame.font.Font(None, 24)
            text_surface = font.render(self.text, True, BLACK)
            text_rect = text_surface.get_rect(center=self.rect.center)
            screen.blit(text_surface, text_rect) 