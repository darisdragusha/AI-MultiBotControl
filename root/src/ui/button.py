import pygame
from src.core.constants import BLACK, WHITE, DARK_GRAY, LIGHT_GRAY

class Button:
    def __init__(self, x, y, width, height, text, icon=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.icon = icon
        self.selected = False

    def draw(self, screen):
        # Define colors for buttons
        base_color = (200, 200, 200) if not self.selected else (100, 150, 250)  # Light gray or blue
        shadow_color = (150, 150, 150)  # Subtle shadow color
        border_color = DARK_GRAY  # Border color

        # Draw shadow effect
        shadow_offset = 4
        shadow_rect = self.rect.move(shadow_offset, shadow_offset)
        pygame.draw.rect(screen, shadow_color, shadow_rect, border_radius=8)

        # Draw button with rounded corners
        pygame.draw.rect(screen, base_color, self.rect, border_radius=8)

        # Draw border
        pygame.draw.rect(screen, border_color, self.rect, width=2, border_radius=8)

        # Draw icon if provided
        if self.icon:
            screen.blit(self.icon, (self.rect.x + 10, self.rect.centery - self.icon.get_height() // 2))

        # Draw button text
        if self.text:
            font = pygame.font.Font(None, 25)
            text_surface = font.render(self.text, True, BLACK)
            text_rect = text_surface.get_rect(center=self.rect.center)
            screen.blit(text_surface, text_rect)

        # Highlight effect if selected
        if self.selected:
            pygame.draw.rect(screen, LIGHT_GRAY, self.rect, width=2, border_radius=8)
