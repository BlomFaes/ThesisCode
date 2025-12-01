# rocket_evolution_replay.py
# PERFECT FINAL — fuel bars, best on top, adjustable mutation, all rockets visible

import math
import pygame
import random

# ------------------------------
# SETTINGS — CHANGE THESE!
# ------------------------------
WIDTH = 800
HEIGHT = 600
GROUND_Y = HEIGHT - 80
PAD_X = 360
PAD_Y = GROUND_Y - 8
PAD_WIDTH = 80
PAD_CENTER_X = 400

START_X = 200
START_Y = 100
START_VX = 2.0
START_VY = 0.0
START_ANGLE = -10

SAVE_EVERY = 10
POP_SIZE = 60
MAX_STEPS = 1200
FUEL_START = 300.0

MUTATION_RATE = 0.3      # ← CHANGE THIS! (0.1 = low diversity, 0.4 = high)
MUTATION_STRENGTH = 0.6  # ← how strong mutations are

# ------------------------------
# Fast Brain
# ------------------------------
def sigmoid(x):
    return 1 / (1 + math.exp(-max(-500, min(500, x))))

class Brain:
    def __init__(self):
        self.w1 = [[random.uniform(-2,2) for _ in range(8)] for _ in range(16)]
        self.w2 = [[random.uniform(-2,2) for _ in range(16)] for _ in range(3)]

    def predict(self, inp):
        h = [sigmoid(sum(self.w1[i][j] * inp[j] for j in range(8))) for i in range(16)]
        return [sigmoid(sum(self.w2[i][j] * h[j] for j in range(16))) for i in range(3)]

    def copy(self):
        b = Brain()
        b.w1 = [row[:] for row in self.w1]
        b.w2 = [row[:] for row in self.w2]
        return b

    def mutate(self):
        for row in self.w1 + self.w2:
            for i in range(len(row)):
                if random.random() < MUTATION_RATE:
                    row[i] += random.gauss(0, MUTATION_STRENGTH)


# ------------------------------
# Rocket
# ------------------------------
class Rocket:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = START_X
        self.y = START_Y
        self.vx = START_VX
        self.vy = START_VY
        self.angle = START_ANGLE
        self.fuel = FUEL_START
        self.thrust = self.left = self.right = False
        self.crashed = False
        self.landed = False

    def apply_thrust(self):
        if self.fuel <= 0 or self.landed: return
        self.fuel -= 1
        a = math.radians(self.angle)
        self.vx += 0.08 * math.sin(a)
        self.vy -= 0.08 * math.cos(a)

    def rotate_left(self):
        if self.fuel > 0 and not self.landed:
            self.fuel -= 0.25
            self.angle -= 1.8

    def rotate_right(self):
        if self.fuel > 0 and not self.landed:
            self.fuel -= 0.25
            self.angle += 1.8

    def update(self):
        if self.crashed or self.landed: return
        self.vy += 0.05
        self.x += self.vx
        self.y += self.vy

    def bottom_y(self):
        a = math.radians(self.angle)
        c, s = math.cos(a), math.sin(a)
        corners = [(-5,-25),(5,-25),(5,25),(-5,25)]
        return max(self.y + cx*s + cy*c for cx, cy in corners)

    def check_landing(self):
        if self.crashed or self.landed: return
        bottom = self.bottom_y()
        on_pad = PAD_X <= self.x <= PAD_X + PAD_WIDTH
        if on_pad and bottom >= PAD_Y:
            speed = math.hypot(self.vx, self.vy)
            if speed < 1.5 and abs(self.angle) < 9:
                self.landed = True
                self.vx = self.vy = 0
                self.thrust = False
                self.y = PAD_Y - 25
                self.x = max(PAD_X + 10, min(PAD_X + PAD_WIDTH - 10, self.x))
            else:
                self.crashed = True
        elif bottom >= GROUND_Y:
            self.crashed = True

    def corners(self):
        a = math.radians(self.angle)
        c, s = math.cos(a), math.sin(a)
        pts = [(-5,-25),(5,-25),(5,25),(-5,25)]
        return [(self.x + x*c - y*s, self.y + x*s + y*c) for x, y in pts]


# ------------------------------
# Fitness — strong centering
# ------------------------------
def evaluate(brain):
    r = Rocket()
    best = 99999
    for _ in range(MAX_STEPS):
        dx = r.x - PAD_CENTER_X
        dy = r.y - PAD_Y
        dist = math.hypot(dx, dy)
        horiz_dist = abs(dx)
        speed = math.hypot(r.vx, r.vy)

        inputs = [dx/800, dy/600, r.vx/6, r.vy/6, r.angle/90, dist/1000, r.x/800, (800-r.x)/800]
        out = brain.predict(inputs)
        r.thrust, r.left, r.right = [x>0.5 for x in out]

        if r.thrust: r.apply_thrust()
        if r.left:  r.rotate_left()
        if r.right: r.rotate_right()
        r.update()
        r.check_landing()

        penalty = dist*3 + horiz_dist*6 + abs(r.angle)*10 + speed*50 + max(0,speed-1.5)*300 + max(0,abs(r.angle)-8)*800
        if dist < 50 and speed < 1.3 and abs(r.angle) < 8:
            penalty -= 3000
        best = min(best, penalty)

        if r.landed:
            return 300000 + r.fuel*60 + 10000*(1 - horiz_dist/40)
        if r.crashed or abs(r.x) > 2000:
            return 5000 - best*3
    return 5000 - best*3


# ------------------------------
# Record full generation
# ------------------------------
def record_generation(population):
    rockets = [Rocket() for _ in population]
    history = [[] for _ in rockets]

    for step in range(MAX_STEPS):
        all_done = True
        for i, (r, brain) in enumerate(zip(rockets, population)):
            if r.crashed or (r.landed and len(history[i]) > 60):
                history[i].append(history[i][-1])  # repeat last frame
                continue

            if r.crashed or r.landed:
                history[i].append(history[i][-1])
                continue

            all_done = False
            dx = r.x - PAD_CENTER_X
            dy = r.y - PAD_Y
            dist = math.hypot(dx, dy)

            inputs = [dx/800, dy/600, r.vx/6, r.vy/6, r.angle/90, dist/1000, r.x/800, (800-r.x)/800]
            out = brain.predict(inputs)
            r.thrust, r.left, r.right = [x>0.5 for x in out]

            if r.thrust: r.apply_thrust()
            if r.left:  r.rotate_left()
            if r.right: r.rotate_right()
            r.update()
            r.check_landing()

            history[i].append((r.x, r.y, r.angle, r.fuel, r.thrust, r.landed, r.crashed))

        if all_done:
            break

    return history


# ------------------------------
# Replay — ALL rockets, fuel bars, best on top
# ------------------------------
def replay_generation(history, gen, is_final=False):
    if not history: return
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)

    max_steps = max(len(h) for h in history)
    i = 0
    while i < max_steps or is_final:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    i = 0
                if event.key == pygame.K_RIGHT and not is_final:
                    pygame.quit()
                    return

        screen.fill((135,206,235))
        pygame.draw.rect(screen, (34,139,34), (0, GROUND_Y, WIDTH, 100))
        pygame.draw.rect(screen, (100,100,100), (PAD_X, PAD_Y, PAD_WIDTH, 8))

        # Draw ALL rockets except the best first
        for idx in range(1, len(history)):  # skip best (idx 0)
            if i >= len(history[idx]): continue
            x, y, a, f, thrust, landed, crashed = history[idx][i]
            if crashed: continue

            corners = Rocket()
            corners.x, corners.y, corners.angle = x, y, a
            pygame.draw.polygon(screen, (255,255,255), corners.corners())

            # Fuel bar
            frac = f / FUEL_START
            h = 44 * frac
            ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
            poly = [(-4,22-h),(4,22-h),(4,22),(-4,22)]
            fuel_poly = [(x + lx*ca - ly*sa, y + lx*sa + ly*ca) for lx, ly in poly]
            col = (0,200,0) if frac>0.6 else (220,180,0) if frac>0.25 else (200,40,0)
            pygame.draw.polygon(screen, col, fuel_poly)

            if thrust and f > 0 and not landed:
                fx = x + 28 * math.sin(math.radians(-a))
                fy = y + 28 * math.cos(math.radians(-a))
                pygame.draw.circle(screen, (255,180,0), (int(fx), int(fy)), 9)

        # Draw BEST rocket LAST → on top
        if i < len(history[0]):
            x, y, a, f, thrust, landed, crashed = history[0][i]
            if not crashed:
                corners = Rocket()
                corners.x, corners.y, corners.angle = x, y, a
                pygame.draw.polygon(screen, (0,255,0), corners.corners())  # GREEN

                # Fuel bar for best
                frac = f / FUEL_START
                h = 44 * frac
                ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
                poly = [(-4,22-h),(4,22-h),(4,22),(-4,22)]
                fuel_poly = [(x + lx*ca - ly*sa, y + lx*sa + ly*ca) for lx, ly in poly]
                col = (0,255,0) if frac>0.6 else (255,255,0) if frac>0.25 else (255,100,0)
                pygame.draw.polygon(screen, col, fuel_poly)

                if thrust and f > 0 and not landed:
                    fx = x + 28 * math.sin(math.radians(-a))
                    fy = y + 28 * math.cos(math.radians(-a))
                    pygame.draw.circle(screen, (255,220,0), (int(fx), int(fy)), 11)

        title = f"PERFECT LANDING! Gen {gen}" if is_final else f"Generation {gen} — All Rockets"
        txt1 = font.render(title, True, (0,0,0))
        txt2 = font.render("SPACE = restart    Right Arrow = next    ESC = quit", True, (0,0,0))
        screen.blit(txt1, (10,10))
        screen.blit(txt2, (10,560))

        pygame.display.flip()
        clock.tick(60)
        i += 1


# ------------------------------
# Evolution
# ------------------------------
population = [Brain() for _ in range(POP_SIZE)]
saved_generations = []
gen = 0

print(f"Training... (Mutation rate: {MUTATION_RATE}, Strength: {MUTATION_STRENGTH})")

while True:
    gen += 1
    scores = [evaluate(brain) for brain in population]
    sorted_idx = sorted(range(POP_SIZE), key=lambda i: scores[i], reverse=True)
    population = [population[i] for i in sorted_idx]
    best_score = scores[sorted_idx[0]]

    print(f"Gen {gen} → Best: {best_score:.0f}")

    if gen % SAVE_EVERY == 0:
        print("   → Saving full generation")
        saved_generations.append((gen, record_generation(population)))

    if best_score >= 300000:
        print(f"\nSUCCESS! Perfect landing in generation {gen}!")
        saved_generations.append((gen, record_generation(population)))
        break

    elites = [population[i].copy() for i in range(8)]
    next_pop = elites[:]
    while len(next_pop) < POP_SIZE:
        child = random.choice(elites).copy()
        child.mutate()
        next_pop.append(child)
    population = next_pop

# Show replays
print(f"\nNow playing {len(saved_generations)} generations...")
for g, hist in saved_generations:
    replay_generation(hist, g, g == saved_generations[-1][0])

print("Done! Enjoy your perfect rocket swarm!")