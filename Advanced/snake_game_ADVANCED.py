import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import time

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
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 20
DESPAWN_TIME = 6

class SnakeGameAI:
    # snake game class
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h

        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()


    def __repr__(self):
        # model name
        return "Advanced"


    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.counter = 0
        self.obstacles = list()
        self.food = None
        self.record = list()
        self.food_last_spawned = None
        self._place_food()
        self._place_obstacles()
        self.frame_iteration = 0


    def _place_obstacles(self):
        # place obstacles
        i=1
        while i!=15:
            x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE

            self.record.append(Point(x - BLOCK_SIZE, y))
            self.record.append(Point(x + BLOCK_SIZE, y))
            self.record.append(Point(x, y - BLOCK_SIZE))
            self.record.append(Point(x, y + BLOCK_SIZE))
            self.record.append(Point(x,y))

            obj = Point(x, y)
            if obj in self.snake:
                continue
            self.obstacles.append(obj)
            i += 1


    def _place_food(self):
        self.food_last_spawned = time.time()
        # place food
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE

        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        if self.food in self.record:
            self._place_food()


    def play_step(self, action, current_time):
        # play one step
        self.frame_iteration += 1
        # collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # move obstacles
        self.counter = (self.counter+1)%16
        if(self.counter == 0):
            for i in range(len(self.obstacles)):
                x, y = self.obstacles[i].x, self.obstacles[i].y
                if i % 2 ==0:
                    obj = Point(x-BLOCK_SIZE, y)
                else:
                    obj = Point(x, y-BLOCK_SIZE)
                self.obstacles[i] = obj
            
        elif(self.counter == 4):
            for i in range(len(self.obstacles)):
                x, y = self.obstacles[i].x, self.obstacles[i].y
                if i % 2 ==0:
                    obj = Point(x+BLOCK_SIZE, y)
                else:
                    obj = Point(x, y+BLOCK_SIZE)
                self.obstacles[i] = obj
        
        elif(self.counter == 8):
            for i in range(len(self.obstacles)):
                x, y = self.obstacles[i].x, self.obstacles[i].y
                if i % 2 ==0:
                    obj = Point(x+BLOCK_SIZE, y)
                else:
                    obj = Point(x, y+BLOCK_SIZE)
                self.obstacles[i] = obj

        elif(self.counter == 12):
            for i in range(len(self.obstacles)):
                x, y = self.obstacles[i].x, self.obstacles[i].y
                if i % 2 ==0:
                    obj = Point(x-BLOCK_SIZE, y)
                else:
                    obj = Point(x, y-BLOCK_SIZE)
                self.obstacles[i] = obj
        
        # move snake
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -20
            return reward, game_over, self.score

        # place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 25
            self._place_food()
        else:
            self.snake.pop()
        
        if current_time - self.food_last_spawned >= DESPAWN_TIME:
            reward = -15
            self._place_food()
        
        # update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # return game over and score
        return reward, game_over, self.score


    def is_collision(self, pt=None):
        # check if snake has collided
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        if pt in self.record:
            return True

        return False


    def _update_ui(self):
        # update user interface
        self.display.fill(BLACK)

        for _ in self.obstacles:
            pygame.draw.rect(self.display, WHITE, pygame.Rect(_.x, _.y, BLOCK_SIZE, BLOCK_SIZE))

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def _move(self, action):
        # allow the agent to move
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)