import pygame
import sys
import random
import csv
import os
from datetime import datetime

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
GRID_SIZE = 100
ROWS = HEIGHT // GRID_SIZE
COLS = WIDTH // GRID_SIZE

# Colors
BROWN = (139, 69, 19)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Cell states
EMPTY = 0
PLANTED = 1
WATERED = 2
READY_TO_HARVEST = 3
DRY = 4

# Setup screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Drone Farming Game")
clock = pygame.time.Clock()

# Paths
SESSION_LOG_FOLDER = "farming data and other\data\session_logs"
SUMMARY_CSV = "farming data and other\data\summary_scores.csv"
os.makedirs(SESSION_LOG_FOLDER, exist_ok=True)

# Grid data
class Cell:
    def __init__(self):
        self.state = EMPTY
        self.dry_timer = 0

    def update(self):
        if self.state == PLANTED:
            self.dry_timer += 1
            if self.dry_timer > 300:
                self.state = DRY
        elif self.state == WATERED:
            self.state = READY_TO_HARVEST


def create_grid():
    return [[Cell() for _ in range(COLS)] for _ in range(ROWS)]

def draw_grid():
    for y in range(ROWS):
        for x in range(COLS):
            cell = grid[y][x]
            rect = pygame.Rect(x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE)
            if cell.state == EMPTY:
                color = BROWN
            elif cell.state == PLANTED:
                color = GREEN
            elif cell.state == WATERED:
                color = BLUE
            elif cell.state == READY_TO_HARVEST:
                color = YELLOW
            elif cell.state == DRY:
                color = (150, 75, 0)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, WHITE, rect, 1)

    # Draw drone
    drone_rect = pygame.Rect(player_x * GRID_SIZE, player_y * GRID_SIZE, GRID_SIZE, GRID_SIZE)
    pygame.draw.rect(screen, BLACK, drone_rect)

    # Draw steps remaining
    font = pygame.font.SysFont(None, 36)
    steps_text = font.render(f"Steps left: {steps_remaining}", True, BLACK)
    screen.blit(steps_text, (10, 10))

# Key press cooldown
key_cooldown = 200  # milliseconds
last_key_time = 0

# Logging and saving functions
def save_session_log(session_name, memory_buffer):
    log_path = os.path.join(SESSION_LOG_FOLDER, session_name + ".csv")
    with open(log_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Action",  "x_pos" ,"y_pos",  "time"])
        for action in memory_buffer:
            writer.writerow([action[0], action[1], action[2], action[3]])
    return log_path

def append_summary(score, session_path):
    file_exists = os.path.isfile(SUMMARY_CSV)
    with open(SUMMARY_CSV, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Score", "Session File"])
        writer.writerow([score, session_path])

# Main loop
running = True
while running:
    # Init game state
    grid = create_grid()
    player_x = random.randint(0, COLS - 1)
    player_y = random.randint(0, ROWS - 1)
    score = 0
    memory_buffer = []
    steps_remaining = 10
    session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")

    game_active = True
    while game_active:
        screen.fill(WHITE)
        current_time = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                game_active = False

        keys = pygame.key.get_pressed()

        if steps_remaining <= 0:
            game_active = False
            break

        step_hepend = 10 - steps_remaining

        if current_time - last_key_time >= key_cooldown:
            if keys[pygame.K_w] and player_y > 0:
                player_y -= 1
                steps_remaining -= 1
                memory_buffer.append([1, player_x, player_y, step_hepend])
                last_key_time = current_time
            elif keys[pygame.K_s] and player_y < ROWS - 1:
                player_y += 1
                steps_remaining -= 1
                memory_buffer.append([2, player_x, player_y, step_hepend])
                last_key_time = current_time
            elif keys[pygame.K_a] and player_x > 0:
                player_x -= 1
                steps_remaining -= 1
                memory_buffer.append([3, player_x, player_y, step_hepend])
                last_key_time = current_time
            elif keys[pygame.K_d] and player_x < COLS - 1:
                player_x += 1
                steps_remaining -= 1
                memory_buffer.append([4, player_x, player_y, step_hepend])
                last_key_time = current_time

            cell = grid[player_y][player_x]

            if keys[pygame.K_e]:
                if cell.state == EMPTY:
                    cell.state = PLANTED
                    cell.dry_timer = 0
                    score += 1
                    steps_remaining -= 1
                    memory_buffer.append([5, player_x, player_y, step_hepend])
                    last_key_time = current_time

            elif keys[pygame.K_q]:
                if cell.state == PLANTED:
                    cell.state = WATERED
                    cell.dry_timer = 0
                    score += 1
                    steps_remaining -= 1
                    memory_buffer.append([6, player_x, player_y, step_hepend])
                    last_key_time = current_time

            elif keys[pygame.K_SPACE]:
                if cell.state == READY_TO_HARVEST:
                    cell.state = EMPTY
                    score += 5
                    steps_remaining -= 1
                    memory_buffer.append([0, player_x, player_y, step_hepend])
                    last_key_time = current_time

        for row in grid:
            for c in row:
                c.update()

        for row in grid:
            for c in row:
                if c.state == DRY:
                    score -= 5
                    c.state = EMPTY
                    memory_buffer.append([1, player_x, player_y, step_hepend])

        draw_grid()
        pygame.display.flip()
        clock.tick(60)

    # Save session
    session_file_path = save_session_log(session_name, memory_buffer)
    append_summary(score, session_file_path)

pygame.quit()
sys.exit()