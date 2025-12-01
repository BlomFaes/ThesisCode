# rocket_evolution_final_fixed.py
# FINAL — no syntax errors, perfect behavior

import math
import pygame
import random

# ------------------------------
# SETTINGS
# ------------------------------
WIDTH = 800
HEIGHT = 600
GROUND_Y = HEIGHT - 80
PAD_X = 360
PAD_Y = GROUND_Y - 8
PAD_WIDTH = 80
PAD_CENTER_X = 400

START_X = 400
START_Y = 100
START_VX = 0
START_VY = 0
START_ANGLE = 0

SAVE_EVERY = 10
POP_SIZE = 50
MAX_STEPS = 1200
FUEL_START = 200.0

MUTATION_RATE = 0.35
MUTATION_STRENGTH = 0.6

# ------------------------------
# Brain
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
        b = Brain(); b.w1 = [r[:] for r in self.w1]; b.w2 = [r[:] for r in self.w2]; return b
    def mutate(self):
        for row in self.w1 + self.w2:
            for i in range(len(row)):
                if random.random() < MUTATION_RATE:
                    row[i] += random.gauss(0, MUTATION_STRENGTH)

# ------------------------------
# Rocket — FIXED rotate methods
# ------------------------------
class Rocket:
    def __init__(self): self.reset()
    def reset(self):
        self.x = START_X; self.y = START_Y; self.vx = START_VX; self.vy = START_VY
        self.angle = START_ANGLE; self.fuel = FUEL_START
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
        pad_left  = PAD_X + 8
        pad_right = PAD_X + PAD_WIDTH - 8
        if pad_left <= self.x <= pad_right and bottom >= PAD_Y - 3:
            speed = math.hypot(self.vx, self.vy)
            if speed < 0.1 and abs(self.angle) < 4 and abs(self.vx) < 0.7:
                self.landed = True
                self.vx = self.vy = 0
                self.thrust = False
                self.y = PAD_Y - 25
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
# Fitness & Record (unchanged)
# ------------------------------
def evaluate(brain):
    r = Rocket()
    best_penalty = 999999
    for _ in range(MAX_STEPS):
        dx = r.x - PAD_CENTER_X
        dist = math.hypot(dx, r.y - PAD_Y)
        inputs = [dx/800, (r.y-PAD_Y)/600, r.vx/6, r.vy/6, r.angle/90, dist/1000, r.x/800, (800-r.x)/800]
        out = brain.predict(inputs)
        r.thrust, r.left, r.right = [x>0.5 for x in out]
        if r.thrust: r.apply_thrust()
        if r.left:  r.rotate_left()
        if r.right: r.rotate_right()
        r.update()
        r.check_landing()

        penalty = dist*5 + abs(dx)*10 + abs(r.angle)*15 + math.hypot(r.vx,r.vy)*70 + abs(r.vx)*150
        best_penalty = min(best_penalty, penalty)

        if r.landed:
            return 600000 + r.fuel*100
        if r.crashed or abs(r.x) > 2000:
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
            inputs = [dx/800, dy/600, r.vx/6, r.vy/6, r.angle/90,
                      math.hypot(dx,dy)/1000, r.x/800, (800-r.x)/800]
            out = brain.predict(inputs)
            r.thrust, r.left, r.right = [x>0.5 for x in out]
            if r.thrust: r.apply_thrust()
            if r.left: r.rotate_left()
            if r.right: r.rotate_right()
            r.update()
            r.check_landing()
            history[i].append((r.x, r.y, r.angle, r.fuel, r.thrust, r.landed, r.crashed))
        if done: break
    return history

# ------------------------------
# Replay — final = only best, clean look
# ------------------------------
def replay_generation(history, gen, is_final=False):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 42)
    max_steps = max(len(h) for h in history)
    i = 0

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit(); return False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE: i = 0
                if e.key == pygame.K_RIGHT: return True

        screen.fill((135,206,235))
        pygame.draw.rect(screen, (34,139,34), (0, GROUND_Y, WIDTH, 100))
        pygame.draw.rect(screen, (90,90,90), (PAD_X, PAD_Y, PAD_WIDTH, 8))

        if is_final:
            # FINAL: only best rocket, classic clean white
            if i < len(history[0]):
                x,y,a,f,thrust,landed,crashed = history[0][i]
                if not crashed:
                    corners = Rocket(); corners.x,corners.y,corners.angle = x,y,a
                    pygame.draw.polygon(screen, (230,230,230), corners.corners())
                    pygame.draw.polygon(screen, (150,150,150), corners.corners(), 3)

                    frac = f/FUEL_START; h = 44*frac
                    ca,sa = math.cos(math.radians(a)), math.sin(math.radians(a))
                    poly = [(-4,22-h),(4,22-h),(4,22),(-4,22)]
                    fuel_poly = [(x + lx*ca - ly*sa, y + lx*sa + ly*ca) for lx,ly in poly]
                    col = (0,220,0) if frac>0.6 else (220,180,0) if frac>0.25 else (220,50,50)
                    pygame.draw.polygon(screen, col, fuel_poly)

                    if thrust and f>0:
                        fx = x + 30*math.sin(math.radians(-a))
                        fy = y + 30*math.cos(math.radians(-a))
                        pygame.draw.circle(screen, (255,200,50), (int(fx),int(fy)), 14)

            title = "PERFECT LANDING!"
            txt = font.render(title, True, (255,255,120))
            shadow = font.render(title, True, (100,60,0))
            screen.blit(shadow, (202,152)); screen.blit(txt, (200,150))
        else:
            # NORMAL generations: full swarm + cyan best
            for idx in range(1, len(history)):
                if i >= len(history[idx]): continue
                x,y,a,f,thrust,_,crashed = history[idx][i]
                if crashed: continue
                corners = Rocket(); corners.x,corners.y,corners.angle = x,y,a
                pygame.draw.polygon(screen, (255,255,255), corners.corners())
                frac = f/FUEL_START; h = 44*frac
                ca,sa = math.cos(math.radians(a)), math.sin(math.radians(a))
                poly = [(-4,22-h),(4,22-h),(4,22),(-4,22)]
                fuel_poly = [(x + lx*ca - ly*sa, y + lx*sa + ly*ca) for lx,ly in poly]
                col = (0,220,0) if frac>0.6 else (220,180,0) if frac>0.25 else (220,50,50)
                pygame.draw.polygon(screen, col, fuel_poly)
                if thrust and f>0:
                    fx = x + 28*math.sin(math.radians(-a))
                    fy = y + 28*math.cos(math.radians(-a))
                    pygame.draw.circle(screen, (255,180,0), (int(fx),int(fy)), 9)

            if i < len(history[0]):
                x,y,a,f,thrust,landed,crashed = history[0][i]
                if not crashed:
                    corners = Rocket(); corners.x,corners.y,corners.angle = x,y,a
                    pygame.draw.polygon(screen, (0,255,255), corners.corners())
                    pygame.draw.polygon(screen, (0,180,220), corners.corners(), 4)
                    frac = f/FUEL_START; h = 44*frac
                    ca,sa = math.cos(math.radians(a)), math.sin(math.radians(a))
                    poly = [(-4,22-h),(4,22-h),(4,22),(-4,22)]
                    fuel_poly = [(x + lx*ca - ly*sa, y + lx*sa + ly*ca) for lx,ly in poly]
                    col = (0,255,255) if frac>0.6 else (0,220,255) if frac>0.25 else (0,180,220)
                    pygame.draw.polygon(screen, col, fuel_poly)
                    if thrust and f>0:
                        fx = x + 30*math.sin(math.radians(-a))
                        fy = y + 30*math.cos(math.radians(-a))
                        pygame.draw.circle(screen, (255,255,100), (int(fx),int(fy)), 13)

            title = f"Generation {gen}"

        # Text overlay
        txt = font.render(title, True, (255,255,255))
        shadow = font.render(title, True, (0,0,0))
        instr = font.render("SPACE = restart    Right Arrow = next    ESC = quit", True, (255,255,255))
        instr_s = font.render("SPACE = restart    Right Arrow = next    ESC = quit", True, (0,0,0))
        screen.blit(shadow, (12,12)); screen.blit(txt, (10,10))
        screen.blit(instr_s, (12,562)); screen.blit(instr, (10,560))

        pygame.display.flip()
        clock.tick(60)
        i += 1
        if i >= max_steps: i = max_steps - 1

# ------------------------------
# Evolution
# ------------------------------
population = [Brain() for _ in range(POP_SIZE)]
saved = []
gen = 0
print("Training with realistic landings...")

while True:
    gen += 1
    scores = [evaluate(b) for b in population]
    idx = sorted(range(POP_SIZE), key=lambda i: scores[i], reverse=True)
    population = [population[i] for i in idx]
    best = scores[idx[0]]
    print(f"Gen {gen} → Best: {best:.0f}")

    if gen % SAVE_EVERY == 0:
        print("   → Saved")
        saved.append((gen, record_generation(population)))

    if best >= 600000:
        print(f"\nPERFECT LANDING IN GEN {gen}!")
        saved.append((gen, record_generation(population)))
        break

    elites = [p.copy() for p in population[:8]]
    next_pop = elites[:]
    while len(next_pop) < POP_SIZE:
        child = random.choice(elites).copy()
        child.mutate()
        next_pop.append(child)
    population = next_pop

# ------------------------------
# Replays
# ------------------------------
print(f"\nPlaying {len(saved)} replays — final shows only the champion in clean style")
for i, (g, h) in enumerate(saved):
    replay_generation(h, g, i == len(saved)-1)

print("Done! Enjoy your masterpiece.")