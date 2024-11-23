import torch
import random
import numpy as np
from collections import deque
from snake_game_EXPERT import SnakeGameAI, Direction, Point
from dqn_network import Linear_QNet, QTrainer
from helper import plot
import time
import os

BLOCK_SIZE = 20 

# Hyperparameters
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
GAMMA = 0.9  # Discount rate
EPSILON_START = 80  # Exploration rate start

class Agent:
    def __init__(self, model_name="Expert"):
        self.n_games = 0
        self.epsilon = EPSILON_START  # Exploration rate
        self.gamma = GAMMA  # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # Popleft() if full
        self.model = Linear_QNet(11, 256, 3)  # Input 11, Hidden 256, Output 3
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.model_name = model_name

        # Load the best model if exists
        model_path = f"{model_name} Model.pth"
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, weights_only=True))
            print(f"{model_name} model loaded successfully.")
        else:
            print(f"{model_name} model file not found. Initializing with default weights.")

    def get_state(self, game, snake_id):
        if snake_id == 1:
            head = game.snake1[0]
            direction = game.direction1
            snake = game.snake1
            
        else:
            head = game.snake2[0]
            direction = game.direction2
            snake = game.snake2

        # Points around the snake's head
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        # Current direction
        dir_l = direction == Direction.LEFT
        dir_r = direction == Direction.RIGHT
        dir_u = direction == Direction.UP
        dir_d = direction == Direction.DOWN

        # Danger states
        danger_straight = (dir_r and game._is_collision(point_r, snake)) or \
                          (dir_l and game._is_collision(point_l, snake)) or \
                          (dir_u and game._is_collision(point_u, snake)) or \
                          (dir_d and game._is_collision(point_d, snake))

        danger_right = (dir_u and game._is_collision(point_r, snake)) or \
                       (dir_d and game._is_collision(point_l, snake)) or \
                       (dir_l and game._is_collision(point_u, snake)) or \
                       (dir_r and game._is_collision(point_d, snake))

        danger_left = (dir_d and game._is_collision(point_r, snake)) or \
                      (dir_u and game._is_collision(point_l, snake)) or \
                      (dir_r and game._is_collision(point_u, snake)) or \
                      (dir_l and game._is_collision(point_d, snake))

        # Food priorities
        food = game.food if game.special_food is None else game.special_food
        food_left = food.x < head.x
        food_right = food.x > head.x
        food_up = food.y < head.y
        food_down = food.y > head.y

        # Return the state
        state = [
            danger_straight,
            danger_right,
            danger_left,
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            food_left,
            food_right,
            food_up,
            food_down,
        ]
        return np.array(state)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = EPSILON_START - self.n_games  # Decay exploration
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

def train():
    plot_scores1, plot_scores2 = [], []
    plot_mean_scores1, plot_mean_scores2 = [], []
    total_score1, total_score2 = 0, 0
    record1, record2 = 0, 0

    agent1 = Agent(model_name="Agent1")
    agent2 = Agent(model_name="Agent2")
    game = SnakeGameAI()

    while True:
        current_time = time.time()

        # Get states
        state1 = agent1.get_state(game, snake_id=1)
        state2 = agent2.get_state(game, snake_id=2)

        # Get actions
        action1 = agent1.get_action(state1)
        action2 = agent2.get_action(state2)

        # Perform actions and check game state
        reward1, reward2, done, score1, score2 = game.play_step(action1, action2, current_time)

        # Get new states
        next_state1 = agent1.get_state(game, snake_id=1)
        next_state2 = agent2.get_state(game, snake_id=2)

        # Train agents
        agent1.train_short_memory(state1, action1, reward1, next_state1, done)
        agent2.train_short_memory(state2, action2, reward2, next_state2, done)

        # Remember experiences
        agent1.remember(state1, action1, reward1, next_state1, done)
        agent2.remember(state2, action2, reward2, next_state2, done)

        # If the game is over, either agent has died
        if done:
            game.reset()
            agent1.n_games += 1
            agent2.n_games += 1

            # Train long memory
            agent1.train_long_memory()
            agent2.train_long_memory()

            # Save models only if new records are achieved
            if score1 > record1:
                record1 = score1
                agent1.model.save(f"{agent1.model_name}_best.pth")
            if score2 > record2:
                record2 = score2
                agent2.model.save(f"{agent2.model_name}_best.pth")

            print(f"Game {agent1.n_games} | P1: {score1} (Record: {record1}) | P2: {score2} (Record: {record2})")

            # Update plots
            total_score1 += score1
            total_score2 += score2
            mean_score1 = total_score1 / agent1.n_games
            mean_score2 = total_score2 / agent2.n_games
            plot_scores1.append(score1)
            plot_scores2.append(score2)
            plot_mean_scores1.append(mean_score1)
            plot_mean_scores2.append(mean_score2)
            plot(plot_scores1, plot_mean_scores1, plot_scores2, plot_mean_scores2)

if __name__ == "__main__":
    train()
