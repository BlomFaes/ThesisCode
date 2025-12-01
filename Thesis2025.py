import math
import pygame
import random
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# SETTINGS
# ------------------------------
WIDTH = 800
HEIGHT = 800
GROUND_Y = HEIGHT - 80
PAD_X = 360
PAD_Y = GROUND_Y - 8
PAD_WIDTH = 80
PAD_CENTER_X = 400

ROCKET_WIDTH = 10
ROCKET_HEIGHT = 50

# Use these starting conditions to have the rocket come in at an extreme angle with a lot of velocity (the most realistic landing situation)
START_X = 100
START_Y = 50
START_VX = 5
START_VY = 1
START_ANGLE = -80

# Use these starting conditions to start the rocket right above the pad.
# START_X = WIDTH / 2
# START_Y = 50
# START_VX = 0
# START_VY = 0
# START_ANGLE = -90

# Change this number to see more or less generations!
# For example, setting this number to 5 will show you every 5th generation
# Setting it to 1 will show you every single generation etc.
SAVE_EVERY = 10

POP_SIZE = 50
MAX_STEPS = 800
FUEL_START = 350.0

# The mutation rate can also be changed to see what happens
# Remember this is not whether an agent itself will mutate
# It will choose based on 20% if the nodes in each brain should be mutated
MUTATION_RATE = 0.2
MUTATION_STRENGTH = 1

# ------------------------------
# Brain
# ------------------------------
def sigmoid(x):
    #the normalization function
    return 1 / (1 + math.exp(-max(-500, min(500, x))))

class Brain:
    def __init__(self):
        self.input_nodes = 6
        self.hidden_nodes = 16
        self.output_nodes = 3
        self.w1 = [[random.uniform(-2,2) for _ in range(self.input_nodes)] for _ in range(self.hidden_nodes)]
        self.w2 = [[random.uniform(-2,2) for _ in range(self.hidden_nodes)] for _ in range(self.output_nodes)]
    def predict(self, inp):
        h = [sigmoid(sum(self.w1[i][j] * inp[j] for j in range(self.input_nodes))) for i in range(self.hidden_nodes)]
        return [sigmoid(sum(self.w2[i][j] * h[j] for j in range(self.hidden_nodes))) for i in range(self.output_nodes)]
    def copy(self):
        b = Brain(); b.w1 = [r[:] for r in self.w1]; b.w2 = [r[:] for r in self.w2]; return b
    def mutate(self):
        for row in self.w1 + self.w2:
            for i in range(len(row)):
                if random.random() < MUTATION_RATE:
                    row[i] += random.gauss(0, MUTATION_STRENGTH)

# ------------------------------
# Rocket
# ------------------------------
class Rocket:
    def __init__(self): self.reset()
    def reset(self):
        self.x = START_X
        self.y = START_Y
        self.vx = START_VX
        self.vy = START_VY
        self.angle = START_ANGLE
        self.fuel = FUEL_START
        self.thrust = self.left = self.right = False
        self.crashed = self.landed = False

    def apply_thrust(self):
        if self.fuel <= 0 or self.landed: return
        self.fuel -= 1
        a = math.radians(self.angle)
        self.vx += 0.08 * math.sin(a)
        self.vy -= 0.08 * math.cos(a)

    def rotate_left(self):
        if self.fuel > 0 and not self.landed:
            self.fuel -= 1
            self.angle -= 0.8

    def rotate_right(self):
        if self.fuel > 0 and not self.landed:
            self.fuel -= 1
            self.angle += 0.8

    def update(self):
        if self.crashed or self.landed: return
        self.vy += 0.05 #gravity
        self.x += self.vx
        self.y += self.vy

    def bottom_y(self):
        a = math.radians(self.angle)
        c, s = math.cos(a), math.sin(a)
        return max(self.y + cx*s + cy*c for cx, cy in [(-ROCKET_WIDTH/2,-ROCKET_HEIGHT/2),(ROCKET_WIDTH/2,-ROCKET_HEIGHT/2),(ROCKET_WIDTH/2,ROCKET_HEIGHT/2),(-ROCKET_WIDTH/2,ROCKET_HEIGHT/2)])

    def check_landing(self):
        if self.crashed or self.landed: return
        bottom = self.bottom_y()
        pad_left  = PAD_X + 10
        pad_right = PAD_X + PAD_WIDTH - 10

        if pad_left <= self.x <= pad_right and bottom >= PAD_Y:
            speed = math.hypot(self.vx, self.vy)
            #The tight landing conditions:
            #You can play around with these and see it will land very easily when they are increased!
            if speed < 1 and abs(self.angle) < 3 and abs(self.vx) < 0.6:
                self.landed = True
                self.vx = self.vy = 0
                self.thrust = False
                self.y = PAD_Y - (bottom - self.y)
            else:
                self.crashed = True
        elif bottom >= GROUND_Y:
            self.crashed = True

    def corners(self):
        a = math.radians(self.angle)
        c, s = math.cos(a), math.sin(a)
        pts = [(-ROCKET_WIDTH/2,-ROCKET_HEIGHT/2),(ROCKET_WIDTH/2,-ROCKET_HEIGHT/2),(ROCKET_WIDTH/2,ROCKET_HEIGHT/2),(-ROCKET_WIDTH/2,ROCKET_HEIGHT/2)]
        return [(self.x + x*c - y*s, self.y + x*s + y*c) for x, y in pts]

# ------------------------------
# Fitness & Record
# ------------------------------
def evaluate(brain):
    r = Rocket()
    best_penalty = 999999
    for _ in range(MAX_STEPS):
        dx = r.x - PAD_CENTER_X
        dist = math.hypot(dx, r.y - PAD_Y)
        inputs = [ #normalizing the inputs in order to evaluate the neural network
            dx / WIDTH,
            (r.y-PAD_Y) / HEIGHT,
            r.vx / 8,
            r.vy / 8,
            r.angle / 90,
            dist / 1000,
        ]
        out = brain.predict(inputs)
        r.thrust, r.left, r.right = [x>0.5 for x in out]
        if r.thrust: r.apply_thrust()
        if r.left:  r.rotate_left()
        if r.right: r.rotate_right()
        r.update()
        r.check_landing()

        #fitness function
        penalty = dist*5 + abs(dx)*10 + abs(r.angle)*15 + math.hypot(r.vx,r.vy)*70 + abs(r.vx)*150
        best_penalty = min(best_penalty, penalty)

        if r.landed:
            return 10000 + r.fuel
        if r.crashed or abs(r.x) > WIDTH:
            return 10000 - best_penalty*2
    return 10000 - best_penalty*2

def record_generation(population):
    rockets = [Rocket() for _ in population]
    history = [[] for _ in rockets]
    for _ in range(MAX_STEPS):
        done = True
        for i, (r, brain) in enumerate(zip(rockets, population)):
            if r.crashed or r.landed:
                history[i].append(history[i][-1])
                continue
            done = False
            dx = r.x - PAD_CENTER_X
            dy = r.y - PAD_Y
            inputs = [dx/WIDTH, dy/HEIGHT, r.vx/8, r.vy/8, r.angle/90, math.hypot(dx,dy)/1000]
            out = brain.predict(inputs)
            r.thrust, r.left, r.right = [x>0.5 for x in out]
            if r.thrust: r.apply_thrust()
            if r.left: r.rotate_left()
            if r.right: r.rotate_right()
            r.update()
            r.check_landing()
            history[i].append((r.x, r.y, r.angle, r.fuel, r.thrust, r.landed, r.crashed, out))
        if done: break
    return history

# ------------------------------
# BRAIN VISUALIZATION
# ------------------------------
def draw_neural_network(screen, brain, inputs, outputs):
    x0 = 420
    y0 = 80

    ix = x0 + 40
    hx = x0 + 140
    ox = x0 + 260

    input_pos  = [(ix, y0 + 120 + i*65) for i in range(6)]
    hidden_pos = [(hx, y0 + 100 + i*28) for i in range(16)]
    output_pos = [(ox, y0 + 200 + i*120) for i in range(3)]

    # Connections
    for i, (px, py) in enumerate(input_pos):
        for h, (qx, qy) in enumerate(hidden_pos):
            w = brain.w1[h][i]
            col = (0, 200, 0) if w > 0 else (200, 50, 50)
            pygame.draw.line(screen, col, (px, py), (qx, qy), 1)

    for h, (qx, qy) in enumerate(hidden_pos):
        for o, (rx, ry) in enumerate(output_pos):
            w = brain.w2[o][h]
            col = (0, 220, 0) if w > 0 else (220, 70, 70)
            pygame.draw.line(screen, col, (qx, qy), (rx, ry), 1)

    # Input nodes
    for i, (px, py) in enumerate(input_pos):
        val = min(1.0, abs(inputs[i]) * 3)
        intensity = int(255 * val)
        pygame.draw.circle(screen, (intensity, intensity, 0), (px, py), 14)  # black → yellow
        pygame.draw.circle(screen, (255, 255, 255), (px, py), 14, 2)

    # Hidden nodes
    for px, py in hidden_pos:
        pygame.draw.circle(screen, (220,220,255), (px, py), 10)
        pygame.draw.circle(screen, (100,100,200), (px, py), 10, 2)

    # Output nodes
    for i, (px, py) in enumerate(output_pos):
        active = outputs[i] > 0.5
        col = (255,90,90) if i==0 and active else \
              (90,255,90) if i==1 and active else \
              (90,200,255) if i==2 and active else (90,90,90)
        pygame.draw.circle(screen, col, (px, py), 18)
        pygame.draw.circle(screen, (255,255,200), (px, py), 18, 3)

    # Labels
    font = pygame.font.Font(None, 22)
    for i, (px, py) in enumerate(input_pos):
        txt = font.render(["dx","dy","vx","vy","ang","dist"][i], True, (255,255,255))
        screen.blit(txt, (px-45, py-10))
    for i, (px, py) in enumerate(output_pos):
        txt = font.render(["THRUST","LEFT","RIGHT"][i], True, (255,255,255))
        screen.blit(txt, (px, py-30))

# ------------------------------
# Replay
# ------------------------------
def replay_generation(history, gen, is_final=False, best_brain=None):
    #this entire function is pygame logic which saves a replay of a generation and then replays it using a pygame window
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    max_steps = max(len(h) for h in history)
    i = 0

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit(); return False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE: i = 0
                if e.key == pygame.K_RIGHT: return True

        screen.fill((100,200,250))
        pygame.draw.rect(screen, (50,150,50), (0, GROUND_Y, WIDTH, 100))
        pygame.draw.rect(screen, (42,42,42), (PAD_X, PAD_Y, PAD_WIDTH, 8))

        current_inputs = [0.0]*6
        current_outputs = [0.0]*3

        if i < len(history[0]):
            x, y, a, f, thrust, landed, crashed, out = history[0][i]
            current_outputs = out if out is not None else [0,0,0]
            dx = x - PAD_CENTER_X
            dy = y - PAD_Y
            current_inputs = [dx/WIDTH, dy/HEIGHT, history[0][i][2]/8, history[0][i][3]/8, a/90, math.hypot(dx,dy)/1000]

        if is_final:
            if i < len(history[0]):
                x,y,a,f,thrust,landed,crashed,_ = history[0][i]
                if not crashed:
                    corners = Rocket(); corners.x,corners.y,corners.angle = x,y,a
                    pygame.draw.polygon(screen, (255,255,255), corners.corners())
                    frac = f/FUEL_START
                    h = (ROCKET_HEIGHT-6) * frac
                    ca,sa = math.cos(math.radians(a)), math.sin(math.radians(a))
                    poly = [(-(ROCKET_WIDTH/2-1),ROCKET_HEIGHT/2-3-h),(ROCKET_WIDTH/2-1,ROCKET_HEIGHT/2-3-h),(ROCKET_WIDTH/2-1,ROCKET_HEIGHT/2-3),(-(ROCKET_WIDTH/2-1),ROCKET_HEIGHT/2-3)]
                    fuel_poly = [(x + lx*ca - ly*sa, y + lx*sa + ly*ca) for lx,ly in poly]
                    col = (0,220,0) if frac>0.6 else (220,180,0) if frac>0.25 else (220,50,50)
                    pygame.draw.polygon(screen, col, fuel_poly)
                    if thrust and f>0:
                        fx = x + 30*math.sin(math.radians(-a))
                        fy = y + 30*math.cos(math.radians(-a))
                        pygame.draw.circle(screen, (255,200,50), (int(fx),int(fy)), 9)
        else:
            for idx in range(1, len(history)):
                if i >= len(history[idx]): continue
                x,y,a,f,thrust,_,crashed,_ = history[idx][i]
                if crashed: continue
                corners = Rocket(); corners.x,corners.y,corners.angle = x,y,a
                pygame.draw.polygon(screen, (255,255,255), corners.corners())
                frac = f/FUEL_START
                h = (ROCKET_HEIGHT - 6) * frac
                ca,sa = math.cos(math.radians(a)), math.sin(math.radians(a))
                poly = [(-(ROCKET_WIDTH / 2 - 1), ROCKET_HEIGHT / 2 - 3 - h),(ROCKET_WIDTH / 2 - 1, ROCKET_HEIGHT / 2 - 3 - h),(ROCKET_WIDTH / 2 - 1, ROCKET_HEIGHT / 2 - 3),(-(ROCKET_WIDTH / 2 - 1), ROCKET_HEIGHT / 2 - 3)]
                fuel_poly = [(x + lx*ca - ly*sa, y + lx*sa + ly*ca) for lx,ly in poly]
                col = (0,220,0) if frac>0.6 else (220,180,0) if frac>0.25 else (220,50,50)
                pygame.draw.polygon(screen, col, fuel_poly)
                if thrust and f>0:
                    fx = x + 28*math.sin(math.radians(-a))
                    fy = y + 28*math.cos(math.radians(-a))
                    pygame.draw.circle(screen, (255,180,0), (int(fx),int(fy)), 9)

            if i < len(history[0]):
                x,y,a,f,thrust,landed,crashed,_ = history[0][i]
                if not crashed:
                    corners = Rocket(); corners.x,corners.y,corners.angle = x,y,a
                    pygame.draw.polygon(screen, (255,255,255), corners.corners())
                    frac = f/FUEL_START
                    h = (ROCKET_HEIGHT - 6) * frac
                    ca,sa = math.cos(math.radians(a)), math.sin(math.radians(a))
                    poly = [(-(ROCKET_WIDTH / 2 - 1), ROCKET_HEIGHT / 2 - 3 - h),(ROCKET_WIDTH / 2 - 1, ROCKET_HEIGHT / 2 - 3 - h),(ROCKET_WIDTH / 2 - 1, ROCKET_HEIGHT / 2 - 3),(-(ROCKET_WIDTH / 2 - 1), ROCKET_HEIGHT / 2 - 3)]
                    fuel_poly = [(x + lx*ca - ly*sa, y + lx*sa + ly*ca) for lx,ly in poly]
                    col = (0,255,255) if frac>0.6 else (0,220,255) if frac>0.25 else (0,180,220)
                    pygame.draw.polygon(screen, col, fuel_poly)
                    if thrust and f>0:
                        fx = x + 30*math.sin(math.radians(-a))
                        fy = y + 30*math.cos(math.radians(-a))
                        pygame.draw.circle(screen, (255,255,100), (int(fx),int(fy)), 9)

        if best_brain:
            draw_neural_network(screen, best_brain, current_inputs, current_outputs)

        title = f"Generation {gen}" if not is_final else f"Final Landing - Gen {gen}"
        txt = font.render(title, True, (255,255,255))
        shadow = font.render(title, True, (0,0,0))
        screen.blit(shadow, (12,12)); screen.blit(txt, (10,10))

        pygame.display.flip()
        clock.tick(60)
        i += 1
        if i >= max_steps: i = max_steps - 1


def show_final_analysis(saved, fitness_history):
    if not saved:
        return

    # Final successful landing
    final_gen, final_histories, best_brain = saved[-1]
    traj = final_histories[0]

    # Find exact landing step
    landing_step = next((i for i, step in enumerate(traj) if step[5]), len(traj)-1)
    traj = traj[:landing_step + 1]  # cut after landing

    time = np.arange(len(traj))

    altitudes = []
    for step in traj:
        x, y, angle, fuel, thrust, landed, crashed, _ = step
        # Reconstruct bottom of rocket (same logic as in Rocket.bottom_y())
        a = math.radians(angle)
        c, s = math.cos(a), math.sin(a)
        # Four corner offsets from center
        corners_y = [
            y + x_val * s + y_val * c
            for x_val, y_val in [
                (-ROCKET_WIDTH/2, -ROCKET_HEIGHT/2),
                (ROCKET_WIDTH/2,  -ROCKET_HEIGHT/2),
                (ROCKET_WIDTH/2,   ROCKET_HEIGHT/2),
                (-ROCKET_WIDTH/2,  ROCKET_HEIGHT/2)
            ]
        ]
        bottom_y = max(corners_y)  # highest y = lowest point on screen
        altitude_above_pad = max(0, PAD_Y + 8 - bottom_y)  # PAD_Y + 8 = ground level (top of pad)
        altitudes.append(altitude_above_pad)

    speed = [math.hypot(step[2], step[3]) for step in traj]
    thrust = [90 if step[4] else 0 for step in traj]  # visual bar height

    # === Saved fitness scores ===
    saved_gens = [item[0] for item in saved]
    saved_best_scores = [evaluate(brain) for _, _, brain in saved]

    # ====================================================================
    # GRAPH 1:
    # ====================================================================
    plt.figure(figsize=(13, 7))
    ax1 = plt.gca()

    # Altitude — now truly 0 at landing
    ax1.plot(time, altitudes, color='#1f77b4', linewidth=4, label='Altitude above landing pad')
    ax1.set_ylabel('Altitude', fontsize=14, color='#1f77b4')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.set_ylim(0, max(altitudes) * 1.05 if altitudes else HEIGHT)
    ax1.grid(True, alpha=0.3)

    # Speed
    ax2 = ax1.twinx()
    ax2.plot(time, speed, color='crimson', linewidth=4, label='Velocity')
    ax2.set_ylabel('Speed', fontsize=14, color='crimson')
    ax2.tick_params(axis='y', labelcolor='crimson')

    # Thrust
    ax1.fill_between(time, 0, thrust, color='orange', alpha=0.75, step='pre', label='Main Engine Firing')

    # Landing pad zone
    ax1.axhspan(0, 25, color='green', alpha=0.25, label='Landing Pad')

    # Touchdown marker — now at altitude = 0
    ax1.plot(landing_step, 0, 'o', color='gold', markersize=14, markeredgecolor='black', markeredgewidth=2, label='Touchdown')

    ax1.set_title(f'PERFECT LANDING — Generation {final_gen}\n'
                  'Entry burn + Landing Burn',
                  fontsize=17, fontweight='bold', pad=20)
    ax1.set_xlabel('Simulation Step', fontsize=14)

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=12, framealpha=0.95)

    plt.tight_layout()
    plt.show()

    # ====================================================================
    # GRAPH 2:
    # ====================================================================
    plt.figure(figsize=(12, 6))
    plt.plot(saved_gens, saved_best_scores, 'o-', color='limegreen', linewidth=4,
             markersize=9, markerfacecolor='black', markeredgecolor='black')
    plt.axhline(y=10000, color='gold', linestyle='--', linewidth=3, label='Successful Landing')
    plt.title('Neuroevolution of Rocket Landing \nBest Fitness Over Generations',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Generation', fontsize=14)
    plt.ylabel('Best Fitness Score', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

# ------------------------------
# Evolution
# ------------------------------
population = [Brain() for _ in range(POP_SIZE)]
saved = []
gen = 0
fitness_history = []

while True:
    gen += 1
    scores = [evaluate(b) for b in population]
    idx = sorted(range(POP_SIZE), key=lambda i: scores[i], reverse=True)
    population = [population[i] for i in idx]
    best = scores[idx[0]]
    print(f"Gen {gen} → Best: {best:.0f}")
    if gen == 1 or gen % SAVE_EVERY == 0 or best >= 10000:
        fitness_history.append(best)

    if gen % SAVE_EVERY == 0 or gen == 1:
        saved.append((gen, record_generation(population), population[0].copy()))

    if best >= 10000:
        print(f"\nLANDING ACHIEVED - GEN {gen}")
        saved.append((gen, record_generation(population), population[0].copy()))
        break

    elites = [p.copy() for p in population[:10]]
    next_pop = elites[:]
    while len(next_pop) < POP_SIZE:
        child = random.choice(elites).copy()
        child.mutate()
        next_pop.append(child)
    population = next_pop

# ------------------------------
# Replays
# ------------------------------
for i, (g, h, best_brain) in enumerate(saved):
    replay_generation(h, g, i == len(saved)-1, best_brain)

show_final_analysis(saved, fitness_history)