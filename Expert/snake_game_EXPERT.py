import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font(r'resources/arial.ttf', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN = (0, 255, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
YELLOW1 = (255, 255, 0)
YELLOW2 = (255, 174, 66)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 20
IMMUNITY_TIME = 5  # seconds of immunity after eating special food

class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake Game - Competitive')
        self.clock = pygame.time.Clock()
        
        # Initialize obstacles here to avoid AttributeError
        self.obstacles = []
        self.reset()

    def reset(self):
        # Snake 1
        self.snake1 = [
            Point(self.w / 4, self.h / 2),
            Point(self.w / 4 - BLOCK_SIZE, self.h / 2),
            Point(self.w / 4 - 2 * BLOCK_SIZE, self.h / 2)
        ]
        self.direction1 = Direction.RIGHT
        self.score1 = 0
        self.super_power1 = False
        self.super_power1_time = 0

        # Snake 2
        self.snake2 = [
            Point(3 * self.w / 4, self.h / 2),
            Point(3 * self.w / 4 + BLOCK_SIZE, self.h / 2),
            Point(3 * self.w / 4 + 2 * BLOCK_SIZE, self.h / 2)
        ]
        self.direction2 = Direction.LEFT
        self.score2 = 0
        self.super_power2 = False
        self.super_power2_time = 0

        # Food and Special Food
        self.food = None
        self.special_food = None
        self._place_food()
        self._place_special_food()

        # Obstacles
        self.obstacles = []  # Reinitialize obstacles during reset
        self._place_obstacles()

        self.frame_iteration = 0

    def _place_obstacles(self):
        for _ in range(10):  # Place 10 random obstacles
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            obstacle = Point(x, y)
            if obstacle in self.snake1 or obstacle in self.snake2 or obstacle == self.food or obstacle == self.special_food:
                continue
            self.obstacles.append(obstacle)

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake1 or self.food in self.snake2 or self.food in self.obstacles:
            self._place_food()

    def _place_special_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.special_food = Point(x, y)
        if self.special_food in self.snake1 or self.special_food in self.snake2 or self.special_food in self.obstacles or self.special_food == self.food:
            self._place_special_food()

    def play_step(self, action1, action2, current_time):
        self.frame_iteration += 1

        # Move snakes
        self._move(action1, 1)
        self._move(action2, 2)

        self.snake1.insert(0, self.head1)
        self.snake2.insert(0, self.head2)

        # Collision checks
        reward1, reward2 = 0, 0
        game_over = False

        # Handle immunity timeout
        if self.super_power1 and current_time - self.super_power1_time > IMMUNITY_TIME:
            self.super_power1 = False
        if self.super_power2 and current_time - self.super_power2_time > IMMUNITY_TIME:
            self.super_power2 = False

        # Check collisions for Snake 1
        if (self._is_collision(self.snake1[0], self.snake1) or
                self._check_snake_collision(self.snake1[0], self.snake2) or
                self._check_obstacle_collision(self.snake1[0])):
            if not self.super_power1:
                game_over = True
                reward1 = -10

        # Check collisions for Snake 2
        if (self._is_collision(self.snake2[0], self.snake2) or
                self._check_snake_collision(self.snake2[0], self.snake1) or
                self._check_obstacle_collision(self.snake2[0])):
            if not self.super_power2:
                game_over = True
                reward2 = -30

        if game_over:
            return reward1, reward2, True, self.score1, self.score2

        # Check food consumption
        if self.head1 == self.food:
            self.score1 += 1
            reward1 = 30
            self._place_food()
        elif self.head1 == self.special_food:
            self.super_power1 = True
            self.super_power1_time = current_time
            reward1 = 15
            self._place_special_food()
        else:
            self.snake1.pop()

        if self.head2 == self.food:
            self.score2 += 1
            reward2 = 20
            self._place_food()
        elif self.head2 == self.special_food:
            self.super_power2 = True
            self.super_power2_time = current_time
            reward2 = 10
            self._place_special_food()
        else:
            self.snake2.pop()

        # Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)

        return reward1, reward2, False, self.score1, self.score2

    def _move(self, action, snake_id):
        if snake_id == 1:
            direction = self.direction1
            head = self.snake1[0]
        else:
            direction = self.direction2
            head = self.snake2[0]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        if snake_id == 1:
            self.direction1 = new_dir
        else:
            self.direction2 = new_dir

        x, y = head.x, head.y
        if new_dir == Direction.RIGHT:
            x += BLOCK_SIZE
        elif new_dir == Direction.LEFT:
            x -= BLOCK_SIZE
        elif new_dir == Direction.DOWN:
            y += BLOCK_SIZE
        elif new_dir == Direction.UP:
            y -= BLOCK_SIZE

        if snake_id == 1:
            self.head1 = Point(x, y)
        else:
            self.head2 = Point(x, y)

    def _is_collision(self, pt, snake):
        if pt.x >= self.w or pt.x < 0 or pt.y >= self.h or pt.y < 0:
            return True
        if pt in snake[1:]:
            return True
        return False

    def _check_snake_collision(self, head, other_snake):
        return head in other_snake

    def _check_obstacle_collision(self, head):
        return head in self.obstacles

    def _update_ui(self):
        self.display.fill(BLACK)

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, GREEN, pygame.Rect(self.special_food.x, self.special_food.y, BLOCK_SIZE, BLOCK_SIZE))

        for pt in self.obstacles:
            pygame.draw.rect(self.display, WHITE, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))

        for pt in self.snake1:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        for pt in self.snake2:
            pygame.draw.rect(self.display, YELLOW1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, YELLOW2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        text1 = font.render(f"Player 1: {self.score1}", True, WHITE)
        text2 = font.render(f"Player 2: {self.score2}", True, WHITE)
        self.display.blit(text1, (10, 10))
        self.display.blit(text2, (10, 40))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
