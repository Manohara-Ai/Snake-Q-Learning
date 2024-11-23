# Snake Game Reinforcement Learning

This repository is a deep dive into the evolution of the Snake Game using Reinforcement Learning (RL). It features three progressively complex models—Classic, Advanced, and Expert—each exploring unique gameplay mechanics and RL strategies.

---

## Models

### 1. Classic Snake
- A standard Snake game with basic mechanics.
- Introduces obstacles that the snake must avoid to survive.
- Reinforcement learning agent implemented to learn optimal movements.

### 2. Advanced Snake
- Adds dynamic obstacles that move around the game environment.
- Increased difficulty with adaptive gameplay mechanics.
- Special food is introduced, granting temporary immunity upon consumption.

### 3. Expert Snake
- Features multi-agent gameplay with competitive reinforcement learning.
- Two snakes compete to maximize their scores:
  - Eating food.
  - Avoiding collisions with obstacles, walls, and each other.
- Dynamic environment with both static and moving obstacles.

---

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Install required dependencies from `requirements.txt`:
  ```bash
  pip install -r requirements.txt
