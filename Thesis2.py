# rocket_evolution_replay.py
import numpy as np
import math
import pygame
import sys
from copy import deepcopy

# ------------------------------
# Rocket / Environment (self-contained)
# ------------------------------

class Rocket:
    def __init__(self, x, y, width=10, height=50, fuel=300.0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.vx = 2.0
        self.vy = 0.0
        self.angle = -20  # in degrees, 0 = upright

        # Physics constants (tweakable)
        self.gravity = 0.05
        self.thrust_power = 0.08
        self.rotation_speed = 1.5

        # Fuel limit
        self.fuel = fuel
        self.max_fuel = fuel

        # State
        self.main_thruster = False
        self.left_thruster = False
        self.right_thruster = False
        self.crashed = False
        self.landed = False

    def apply_main_thrust(self):
        if self.fuel <= 0:
            return
        self.fuel = max(0.0, self.fuel - 1.0)
        angle_rad = math.radians(self.angle)
        self.vx += self.thrust_power * math.sin(angle_rad)
        self.vy -= self.thrust_power * math.cos(angle_rad)

    def rotate_left(self):
        if self.fuel <= 0:
            return
        self.fuel = max(0.0, self.fuel)
        self.angle -= self.rotation_speed

    def rotate_right(self):
        if self.fuel <= 0:
            return
        self.fuel = max(0.0, self.fuel)
        self.angle += self.rotation_speed

    def update(self):
        if self.crashed or self.landed:
            return
        self.vy += self.gravity
        self.x += self.vx
        self.y += self.vy

    def get_bottom_y(self):
        angle_rad = math.radians(self.angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        hw = self.width / 2
        hh = self.height / 2
        corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        max_y = float('-inf')
        for cx, cy in corners:
            ry = cx * sin_a + cy * cos_a + self.y
            max_y = max(max_y, ry)
        return max_y

    def check_landing(self, pad_x, pad_y, pad_width, pad_height, ground_y):
        if self.crashed or self.landed:
            return
        rocket_bottom = self.get_bottom_y()
        rocket_center_x = self.x
        on_pad = (pad_x <= rocket_center_x <= pad_x + pad_width)
        if on_pad and rocket_bottom >= pad_y:
            speed = math.hypot(self.vx, self.vy)
            angle_ok = abs(self.angle) < 5
            if speed < 1 and angle_ok:
                self.landed = True
                self.vy = 0
                self.vx = 0
                correction = rocket_bottom - pad_y
                self.y -= correction
            else:
                self.crashed = True
        elif rocket_bottom >= ground_y:
            self.crashed = True

    def get_corners(self):
        angle_rad = math.radians(self.angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        hw = self.width / 2
        hh = self.height / 2
        corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        rotated = []
        for cx, cy in corners:
            rx = cx * cos_a - cy * sin_a + self.x
            ry = cx * sin_a + cy * cos_a + self.y
            rotated.append((rx, ry))
        return rotated

# MultiRocket environment for visualization and headless eval
class MultiRocketEnv:
    def __init__(self, num_rockets=1, width=800, height=600, spawn_x=None, spawn_y=None, initial_fuel=300.0):
        self.width = width
        self.height = height
        self.num_rockets = num_rockets
        self.spawn_x = spawn_x if spawn_x is not None else width // 2
        self.spawn_y = spawn_y if spawn_y is not None else 100
        self.ground_y = height - 80
        self.pad_width = 80
        self.pad_height = 8
        self.pad_x = (width - self.pad_width) // 2
        self.pad_y = self.ground_y - self.pad_height
        self.initial_fuel = initial_fuel
        self.rockets = []
        self.reset()

    def reset(self):
        self.rockets = []
        for _ in range(self.num_rockets):
            r = Rocket(self.spawn_x, self.spawn_y, width=10, height=50, fuel=self.initial_fuel)
            self.rockets.append(r)

    def step_all(self, actions_list):
        for rocket, actions in zip(self.rockets, actions_list):
            if rocket.crashed or rocket.landed:
                continue
            rocket.main_thruster = actions.get('thrust', False)
            rocket.left_thruster = actions.get('left', False)
            rocket.right_thruster = actions.get('right', False)
            if rocket.main_thruster:
                rocket.apply_main_thrust()
            if rocket.left_thruster:
                rocket.rotate_left()
            if rocket.right_thruster:
                rocket.rotate_right()
            rocket.update()
            rocket.check_landing(self.pad_x, self.pad_y, self.pad_width, self.pad_height, self.ground_y)

# ------------------------------
# Simple NN for neuroevolution
# ------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class SimpleNN:
    def __init__(self, input_size=8, hidden_size=10, output_size=3):
        self.w_ih = np.random.randn(hidden_size, input_size) * 0.5
        self.w_ho = np.random.randn(output_size, hidden_size) * 0.5
        self.b_h = np.random.randn(hidden_size, 1) * 0.5
        self.b_o = np.random.randn(output_size, 1) * 0.5

    def activate(self, inputs):
        x = np.array(inputs).reshape(-1, 1)
        h = sigmoid(np.dot(self.w_ih, x) + self.b_h)
        o = sigmoid(np.dot(self.w_ho, h) + self.b_o)
        return o.flatten()

    def mutate(self, rate=0.1, power=0.2):
        for mat in [self.w_ih, self.w_ho, self.b_h, self.b_o]:
            mask = np.random.rand(*mat.shape) < rate
            mat += mask * (np.random.randn(*mat.shape) * power)

    def copy(self):
        return deepcopy(self)

# ------------------------------
# Fitness evaluation (headless)
# ------------------------------
def evaluate_network(nn, max_steps=600, width=800, height=600, spawn_x=None, spawn_y=None, initial_fuel=300.0):
    env = MultiRocketEnv(num_rockets=1, width=width, height=height, spawn_x=spawn_x, spawn_y=spawn_y, initial_fuel=initial_fuel)
    rocket = env.rockets[0]
    pad_center_x = env.pad_x + env.pad_width / 2
    pad_center_y = env.pad_y

    initial_distance = math.hypot(rocket.x - pad_center_x, rocket.y - pad_center_y)
    best_distance = initial_distance

    for step in range(max_steps):
        x_rel = (rocket.x - pad_center_x) / width
        y_rel = (rocket.y - pad_center_y) / height
        vx = rocket.vx / 5
        vy = rocket.vy / 5
        angle = rocket.angle / 180
        distance = math.hypot(rocket.x - pad_center_x, rocket.y - pad_center_y) / width
        dist_to_left = rocket.x / width
        dist_to_right = (width - rocket.x) / width
        inputs = [x_rel, y_rel, vx, vy, angle, distance, dist_to_left, dist_to_right]
        outputs = nn.activate(inputs)
        actions = {'thrust': outputs[0] > 0.5, 'left': outputs[1] > 0.5, 'right': outputs[2] > 0.5}
        env.step_all([actions])

        current_distance = math.hypot(rocket.x - pad_center_x, rocket.y - pad_center_y)
        best_distance = min(best_distance, current_distance)

        if rocket.crashed:
            angle_penalty = abs(rocket.angle)
            fitness = max(0, 1000 - best_distance - angle_penalty * 2)
            return fitness
        if rocket.landed:
            return 10000 + rocket.fuel * 10
        if rocket.x < -50 or rocket.x > width + 50 or rocket.y < -50 or rocket.y > height + 50:
            angle_penalty = abs(rocket.angle)
            return max(0, 1000 - best_distance - angle_penalty * 2)
    return max(0, 1000 - best_distance)

# ------------------------------
# Visualization (records landed histories)
# ------------------------------
def visualize_generation(population, gen, width=800, height=600, max_steps=600, spawn_x=None, spawn_y=None, initial_fuel=300.0):
    """
    Visualize top-N population; record histories of any rockets that land.
    Returns a tuple: (status, landed_histories)
      status: 'quit' -> user canceled/closed window
              'no_landing' -> visualization completed, no landings
              'landing_found' -> one or more rockets landed (histories returned)
      landed_histories: dict mapping rocket_index -> list_of_states
         state = (x,y,angle,vx,vy,fuel,main,left,right)
    """
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(f"Generation {gen} - Top {len(population)}")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 20)

    env = MultiRocketEnv(num_rockets=len(population), width=width, height=height, spawn_x=spawn_x, spawn_y=spawn_y, initial_fuel=initial_fuel)

    pad_center_x = env.pad_x + env.pad_width / 2
    pad_center_y = env.pad_y

    # prepare history lists per rocket
    histories = [[] for _ in range(len(population))]
    landed_histories = {}  # idx -> history

    step = 0
    running = True
    paused = False

    while running and step < max_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return ('quit', {})
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return ('quit', {})

        if not paused:
            actions_list = []
            for idx, (rocket, nn) in enumerate(zip(env.rockets, population)):
                if rocket.crashed or rocket.landed:
                    actions_list.append({})
                    # still append current frame to history (final state)
                    histories[idx].append((rocket.x, rocket.y, rocket.angle, rocket.vx, rocket.vy, rocket.fuel, rocket.main_thruster, rocket.left_thruster, rocket.right_thruster))
                    continue

                # compute inputs
                x_rel = (rocket.x - pad_center_x) / width
                y_rel = (rocket.y - pad_center_y) / height
                vx = rocket.vx / 5
                vy = rocket.vy / 5
                angle = rocket.angle / 180
                distance = math.hypot(rocket.x - pad_center_x, rocket.y - pad_center_y) / width
                dist_to_left = rocket.x / width
                dist_to_right = (width - rocket.x) / width
                inputs = [x_rel, y_rel, vx, vy, angle, distance, dist_to_left, dist_to_right]

                outputs = nn.activate(inputs)
                actions = {'thrust': outputs[0] > 0.5, 'left': outputs[1] > 0.5, 'right': outputs[2] > 0.5}
                actions_list.append(actions)

            env.step_all(actions_list)

            # after stepping, append states to histories and detect landings
            for idx, rocket in enumerate(env.rockets):
                histories[idx].append((rocket.x, rocket.y, rocket.angle, rocket.vx, rocket.vy, rocket.fuel, rocket.main_thruster, rocket.left_thruster, rocket.right_thruster))
                if rocket.landed and idx not in landed_histories:
                    # copy the history of that rocket (landing moment included)
                    landed_histories[idx] = list(histories[idx])

            step += 1

            # Let the generation run to completion even if landings happen,
            # we still record histories of landed rockets above.
            all_done = all(r.crashed or r.landed for r in env.rockets)
            if all_done:
                pygame.time.wait(300)
                break

        # Drawing - crashed rockets are not drawn
        screen.fill((135, 206, 235))
        pygame.draw.rect(screen, (34, 139, 34), (0, env.ground_y, width, height - env.ground_y))
        pygame.draw.rect(screen, (128, 128, 128), (env.pad_x, env.pad_y, env.pad_width, env.pad_height))

        alive_count = 0
        landed_count = 0
        crashed_count = 0

        for rocket in env.rockets:
            if rocket.crashed:
                crashed_count += 1
                continue  # do not draw crashed rockets
            elif rocket.landed:
                landed_count += 1
                color = (100, 255, 100)
            else:
                alive_count += 1
                color = (255, 255, 255)

            # draw rocket body
            corners = rocket.get_corners()
            pygame.draw.polygon(screen, color, corners)

            # draw fuel bar inside rocket (vertical, anchored bottom)
            # compute fuel fraction
            frac = max(0.0, min(1.0, rocket.fuel / rocket.max_fuel))
            # vertical bar dimensions (local unrotated coordinates)
            bar_w = rocket.width * 0.4
            inner_margin = 6
            bar_bottom = (rocket.height / 2) - inner_margin  # local coords
            bar_top = - (rocket.height / 2) + inner_margin
            bar_height_total = bar_bottom - bar_top
            bar_height = bar_height_total * frac
            # local rectangle top/bottom (anchor bottom)
            local_top = bar_bottom - bar_height
            local_bottom = bar_bottom
            # corners of the bar in local coordinates (centered in X)
            local_corners = [
                (-bar_w/2, local_top),
                (bar_w/2, local_top),
                (bar_w/2, local_bottom),
                (-bar_w/2, local_bottom)
            ]
            # rotate and translate fuel bar corners
            angle_rad = math.radians(rocket.angle)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            fuel_poly = []
            for lx, ly in local_corners:
                rx = lx * cos_a - ly * sin_a + rocket.x
                ry = lx * sin_a + ly * cos_a + rocket.y
                fuel_poly.append((rx, ry))
            # color gradient green->yellow->red
            if frac > 0.6:
                fcolor = (0, 200, 0)
            elif frac > 0.25:
                fcolor = (220, 180, 0)
            else:
                fcolor = (200, 40, 0)
            pygame.draw.polygon(screen, fcolor, fuel_poly)

            # draw main thruster flame only if thruster active AND fuel > 0
            if rocket.main_thruster and rocket.fuel > 0 and not rocket.landed:
                angle_rad_vis = math.radians(-rocket.angle)
                offset_x = (rocket.height / 2) * math.sin(angle_rad_vis)
                offset_y = (rocket.height / 2) * math.cos(angle_rad_vis)
                thruster_x = rocket.x + offset_x
                thruster_y = rocket.y + offset_y
                pygame.draw.circle(screen, (255, 255, 0), (int(thruster_x), int(thruster_y)), 5)

        # info text
        info_text = [
            f"Generation: {gen}",
            f"Step: {step}/{max_steps}",
            f"Alive: {alive_count}",
            f"Landed: {landed_count}",
            f"Crashed: {crashed_count}",
            "SPACE: pause/unpause  ESC or close: quit"
        ]
        for i, text in enumerate(info_text):
            surface = font.render(text, True, (0, 0, 0))
            screen.blit(surface, (10, 10 + i * 22))

        if paused:
            pause_text = font.render("PAUSED", True, (255, 0, 0))
            screen.blit(pause_text, (width // 2 - 40, height // 2))

        pygame.display.flip()
        clock.tick(60)

    # After visualizing generation, return status and any landed histories
    if len(landed_histories) > 0:
        pygame.quit()
        return ('landing_found', landed_histories)
    else:
        pygame.quit()
        return ('no_landing', {})

# ------------------------------
# Replay loop (play recorded flight repeatedly)
# ------------------------------
def replay_landing(history, generation_number, width=800, height=600):
    """
    Replay the recorded landing once.
    Press SPACE to replay it from the start.
    ESC or close window exits.
    """

    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Replay - Landed Rocket")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 22)

    # Precompute max fuel from first frame
    if len(history) > 0:
        max_fuel = history[0][5]
        if max_fuel <= 0:
            max_fuel = 1.0
    else:
        max_fuel = 300.0

    idx = 0
    total = len(history)
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break
                elif event.key == pygame.K_SPACE:
                    idx = 0  # restart replay

        screen.fill((135, 206, 235))

        # background and pad
        ground_y = height - 80
        pygame.draw.rect(screen, (34, 139, 34), (0, ground_y, width, height - ground_y))
        pad_width = 80
        pad_height = 8
        pad_x = (width - pad_width) // 2
        pad_y = ground_y - pad_height
        pygame.draw.rect(screen, (128, 128, 128), (pad_x, pad_y, pad_width, pad_height))

        if total == 0:
            msg = font.render("No recorded landing data.", True, (0,0,0))
            screen.blit(msg, (width//2 - 100, height//2))
        else:
            # unpack frame
            x, y, angle, vx, vy, fuel, main, left, right = history[idx]

            # draw rocket (same rendering as before)
            hw = 10/2
            hh = 50/2
            angle_rad = math.radians(angle)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            local = [(-hw,-hh),(hw,-hh),(hw,hh),(-hw,hh)]
            poly=[]
            for lx,ly in local:
                rx = lx*cos_a - ly*sin_a + x
                ry = lx*sin_a + ly*cos_a + y
                poly.append((rx,ry))
            pygame.draw.polygon(screen,(255,255,255),poly)

            # fuel bar (same as before)
            frac = max(0.0, min(1.0, fuel/max_fuel))
            bar_w = 10*0.4
            inner_margin = 6
            bar_bottom = (50/2) - inner_margin
            bar_top = -(50/2) + inner_margin
            bar_total = bar_bottom - bar_top
            bar_h = bar_total * frac
            local_top = bar_bottom - bar_h
            local_bottom = bar_bottom

            fuel_local = [(-bar_w/2,local_top),(bar_w/2,local_top),
                          (bar_w/2,local_bottom),(-bar_w/2,local_bottom)]
            fuel_poly=[]
            for lx,ly in fuel_local:
                rx = lx*cos_a - ly*sin_a + x
                ry = lx*sin_a + ly*cos_a + y
                fuel_poly.append((rx,ry))

            if frac>0.6: fcol=(0,200,0)
            elif frac>0.25: fcol=(220,180,0)
            else: fcol=(200,40,0)
            pygame.draw.polygon(screen,fcol,fuel_poly)

            # flame only when thrust & fuel
            if main and fuel > 0:
                angle_rad_vis = math.radians(-angle)
                ox = (50/2)*math.sin(angle_rad_vis)
                oy = (50/2)*math.cos(angle_rad_vis)
                pygame.draw.circle(screen,(255,255,0),(int(x+ox),int(y+oy)),5)

            # Info text
            lines = [
                f"Replay of generation: {generation_number}",
                f"Frame {idx+1}/{total}",
                "Press SPACE to replay",
                "Press ESC to quit"
            ]
            for i, line in enumerate(lines):
                t = font.render(line, True, (0,0,0))
                screen.blit(t, (10, 10 + i*22))

            # advance frame
            if idx < total - 1:
                idx += 1

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return


# ------------------------------
# Evolution loop (headless, visualize occasionally)
# ------------------------------
def run_evolution():
    # CONFIG
    SPAWN_X = 200  # None = center
    SPAWN_Y = 100
    POP_SIZE = 100
    GENERATIONS = 500
    ELITISM = 1
    VISUALIZE_EVERY = 10
    INITIAL_FUEL = 200.0

    # Initialize population
    population = [SimpleNN() for _ in range(POP_SIZE)]

    for gen in range(GENERATIONS):
        print(f"\n=== GENERATION {gen} ===")

        # Evaluate fitness (headless)
        fitness_scores = []
        for i, nn in enumerate(population):
            fitness = evaluate_network(nn, spawn_x=SPAWN_X, spawn_y=SPAWN_Y, initial_fuel=INITIAL_FUEL)
            fitness_scores.append(fitness)
            if i % 5 == 0:
                print(f"  Eval {i+1}/{POP_SIZE} -> {fitness:.1f}")

        # Sort by fitness (best first)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]

        best_fitness = fitness_scores[0]
        avg_fitness = np.mean(fitness_scores)
        print(f"Best: {best_fitness:.1f} | Avg: {avg_fitness:.1f}")

        # Visualize every Nth generation
        if gen % VISUALIZE_EVERY == 0 or gen == GENERATIONS - 1:
            top_to_show = min(10, POP_SIZE)
            to_visualize = population[:top_to_show]
            status, landed_histories = visualize_generation(to_visualize, gen, spawn_x=SPAWN_X, spawn_y=SPAWN_Y, initial_fuel=INITIAL_FUEL)

            if status == 'quit':
                print("User requested quit during visualization.")
                break

            if status == 'landing_found':
                # At least one rocket landed during this visualization:
                # pick the landed rocket with the smallest index (best among shown)
                landed_indices = sorted(landed_histories.keys())
                chosen_idx = landed_indices[0]
                chosen_history = landed_histories[chosen_idx]
                # corresponding neural net (global population is sorted, and to_visualize are the top N)
                chosen_nn = population[chosen_idx] if chosen_idx < len(population) else population[0]
                print("\nLanding detected in visualization. Stopping evolution and replaying recorded landing.")
                # Replay the recorded landing in loop
                replay_landing(chosen_history, generation_number=gen, width=800, height=600)
                print("Replay finished / user closed window. Exiting.")
                return

        # Elitism & reproduction
        next_gen = [population[i].copy() for i in range(ELITISM)]
        while len(next_gen) < POP_SIZE:
            parent_idx = np.random.randint(0, ELITISM)
            parent = population[parent_idx]
            child = parent.copy()
            child.mutate(rate=0.1, power=0.2)
            next_gen.append(child)

        population = next_gen

    # If evolution finished without an early landing, show best network
    best_nn = population[0]
    print("\nEvolution finished. Showing best network performance (live visualization).")
    visualize_generation([best_nn], gen=GENERATIONS, spawn_x=SPAWN_X, spawn_y=SPAWN_Y, initial_fuel=INITIAL_FUEL)

if __name__ == '__main__':
    try:
        run_evolution()
    finally:
        pygame.quit()
        sys.exit()
