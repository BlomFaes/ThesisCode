import numpy as np
import math
from copy import deepcopy
from Simulation2 import Environment  # replace with your filename

# ------------------------------
# Neural Network Utilities
# ------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class SimpleNN:
    """Small feedforward network with one hidden layer"""
    def __init__(self, input_size=6, hidden_size=6, output_size=3):
        self.w_ih = np.random.randn(hidden_size, input_size)  # Input → Hidden
        self.w_ho = np.random.randn(output_size, hidden_size)  # Hidden → Output
        self.b_h = np.random.randn(hidden_size, 1)
        self.b_o = np.random.randn(output_size, 1)

    def activate(self, inputs):
        x = np.array(inputs).reshape(-1, 1)
        h = sigmoid(np.dot(self.w_ih, x) + self.b_h)
        o = sigmoid(np.dot(self.w_ho, h) + self.b_o)
        return o.flatten()

    def mutate(self, rate=0.1, power=0.5):
        for mat in [self.w_ih, self.w_ho, self.b_h, self.b_o]:
            mutation_mask = np.random.rand(*mat.shape) < rate
            mat += mutation_mask * (np.random.randn(*mat.shape) * power)

    def copy(self):
        return deepcopy(self)

# ------------------------------
# Fitness function
# ------------------------------

def evaluate_network(nn, max_steps=500):
    env = Environment(headless=True)
    rocket = env.rocket
    fitness = 0.0

    for _ in range(max_steps):
        # Normalized inputs
        x_rel = (rocket.x - env.pad_x) / env.width
        y_rel = (rocket.y - env.pad_y) / env.height
        vx = rocket.vx / 10
        vy = rocket.vy / 10
        angle = rocket.angle / 180
        distance = math.hypot(x_rel, y_rel)
        inputs = [x_rel, y_rel, vx, vy, angle, distance]

        # Get outputs
        outputs = nn.activate(inputs)
        actions = {
            'thrust': outputs[0] > 0.5,
            'left': outputs[1] > 0.5,
            'right': outputs[2] > 0.5
        }
        env.step(actions)

        # End early if crashed or landed
        if rocket.crashed:
            fitness -= 100
            break
        if rocket.landed:
            fitness += 1000
            break

        # Reward closeness to pad, low speed, upright orientation
        dist_to_pad = math.hypot(rocket.x - env.pad_x, rocket.y - env.pad_y)
        speed = math.hypot(rocket.vx, rocket.vy)
        fitness += max(0, 10 - dist_to_pad)
        fitness += max(0, 5 - speed)
        fitness += max(0, 2 - abs(rocket.angle)/10)

    return fitness

# ------------------------------
# Evolutionary Loop
# ------------------------------

POP_SIZE = 20
GENERATIONS = 300
ELITISM = 5

# Initialize population
population = [SimpleNN() for _ in range(POP_SIZE)]

for gen in range(GENERATIONS):
    # Evaluate fitness
    fitness_scores = [evaluate_network(nn) for nn in population]
    sorted_indices = np.argsort(fitness_scores)[::-1]
    population = [population[i] for i in sorted_indices]
    best_fitness = fitness_scores[sorted_indices[0]]
    print(f"Generation {gen} | Best fitness: {best_fitness:.2f}")

    # Keep top performers
    next_gen = [population[i].copy() for i in range(ELITISM)]

    # Fill rest with mutated offspring
    while len(next_gen) < POP_SIZE:
        parent = population[np.random.randint(0, ELITISM)]
        child = parent.copy()
        child.mutate(rate=0.2, power=0.3)
        next_gen.append(child)

    population = next_gen

# After evolution, best network
best_nn = population[0]
print("Best network evolved!")

# Example usage (headless)
env = Environment(headless=False)
rocket = env.rocket
running = True
while running:
    # Convert NN outputs to actions
    x_rel = (rocket.x - env.pad_x) / env.width
    y_rel = (rocket.y - env.pad_y) / env.height
    vx = rocket.vx / 10
    vy = rocket.vy / 10
    angle = rocket.angle / 180
    distance = math.hypot(x_rel, y_rel)
    inputs = [x_rel, y_rel, vx, vy, angle, distance]

    outputs = best_nn.activate(inputs)
    actions = {
        'thrust': outputs[0] > 0.5,
        'left': outputs[1] > 0.5,
        'right': outputs[2] > 0.5
    }

    env.step(actions)
    env.render()

    # for event in env.screen.get_events():  # pseudo-code, depends on your pygame loop
    #     if event.type == pygame.QUIT:
    #         running = False
