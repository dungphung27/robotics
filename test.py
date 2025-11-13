import pygame
import sys

# Initialize Pygame
pygame.init()

# Screen settings
screen_width, screen_height = 400, 400  # Increased for scalability
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Scalable Octagon with T")

# Colors
BLACK = (0, 0, 0)
LIGHT_GREEN = (173, 255, 47)
WHITE = (255, 255, 255)

# Draw the octagon with a T
def draw_octagon_with_t(surface, scale):
    # Octagon settings
    cx, cy = screen_width // 2, screen_height // 2  # Center of the octagon
    size = int(70 * scale)  # Base size scaled
    border_thickness = int(8 * scale)  # Border width scaled

    # Define points for the outer octagon
    points_outer = [
        (cx - size // 2, cy - size),        # Top-left
        (cx + size // 2, cy - size),        # Top-right
        (cx + size, cy - size // 2),        # Right-top
        (cx + size, cy + size // 2),        # Right-bottom
        (cx + size // 2, cy + size),        # Bottom-right
        (cx - size // 2, cy + size),        # Bottom-left
        (cx - size, cy + size // 2),        # Left-bottom
        (cx - size, cy - size // 2),        # Left-top
    ]

    # Shrink points inward for the inner octagon
    shrink_factor = border_thickness
    points_inner = [
        (x + (cx - x) * shrink_factor // size, y + (cy - y) * shrink_factor // size)
        for x, y in points_outer
    ]

    # Draw black border (outer octagon)
    pygame.draw.polygon(surface, BLACK, points_outer)

    # Draw light green inner octagon
    pygame.draw.polygon(surface, LIGHT_GREEN, points_inner)

    # Draw "T" shape
    t_width = int(15 * scale)
    t_height = int(70 * scale)
    pygame.draw.rect(surface, BLACK, (cx - t_width // 2, cy - t_height, t_width, t_height))  # Vertical bar
    pygame.draw.rect(surface, BLACK, (cx - t_width, cy - t_height, t_width * 2, t_width // 2))  # Horizontal bar

# Main loop
scale = 1.0  # Initial scale
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:  # Increase scale
                scale += 0.1
            elif event.key == pygame.K_DOWN:  # Decrease scale
                scale = max(0.1, scale - 0.1)  # Prevent negative or zero scale

    screen.fill(WHITE)  # Clear the screen with a white background
    draw_octagon_with_t(screen, scale)  # Draw the shape with the current scale
    pygame.display.flip()  # Update the display

pygame.quit()
sys.exit()
