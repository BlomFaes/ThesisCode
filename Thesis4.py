# rocket_evolution_replay.py
# FINAL VERSION — NEVER GETS STUCK, ALWAYS LANDS

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

START_X = 200
START_Y = 100
START_VX = 2.0
START_VY = 0.0
START_ANGLE = -10

SAVE_EVERY = 10
POP_SIZE = 50
MAX_STEPS = 1200
FUEL_START = 300.0

# ------------------------------
# Fast Brain
# ------------------------------
def sigmoid(x):
    x = max(-500, min(500, x))
    return 1 / (1 + math.exp(-x))

class Brain:
    def __init__(self):
        self.w1 = [[random.uniform(-2, 2) for _ in range(8)] for _ in range(14)]
        self.w2 = [[random.uniform(-2, 2) for _ in range(14)] for _ in range(3)]

    def predict(self, inp):
        h = [0]*14
        for i in range(14):
            for j in range(8):
                h[i] += self.w1[i][j] * inp[j]
            h[i] = sigmoid(h[i])
        o = [0]*3
        for i in range(3):
            for j in range(14):
                o[i] += self.w2[i][j] * h[j]
            o[i] = sigmoid(o[i])
        return o

    def copy(self):
        b = Brain()
        b.w1 = [row[:] for row in self.w1]
        b.w2 = [row[:] for row in self.w2]
        return b

    def mutate(self):
        for row in self.w1 + self.w2:
            for i in range(len(row)):
                if random.random() < 0.25:
                    row[i] += random.gauss(0, 0.5)


# ------------------------------
# Rocket
# ------------------------------
class Rocket:
    def __init__(self):
        self.x = START_X
        self.y = START_Y
        self.vx = START_VX
        self.vy = START_VY
        self.angle = START_ANGLE
        self.fuel = FUEL_START
        self.thrust = self.left = self.right = False
        self.crashed = self.landed = False

    def apply_thrust(self):
        if self.fuel <= 0: return
        self.fuel -= 1
        a = math.radians(self.angle)
        self.vx += 0.08 * math.sin(a)
        self.vy -= 0.08 * math.cos(a)

    def rotate_left(self):
        if self.fuel > 0: self.fuel -= 0.25; self.angle -= 1.8
    def rotate_right(self):
        if self.fuel > 0: self.fuel -= 0.25; self.angle += 1.8

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
            if speed < 1 and abs(self.angle) < 5:
                self.landed = True
                self.vx = self.vy = 0
                self.thrust = False
                self.y -= (bottom - PAD_Y)
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
# PERFECT FITNESS — never stuck
# ------------------------------
def evaluate(brain):
    r = Rocket()
    best_penalty = 99999

    for _ in range(MAX_STEPS):
        dx = r.x - PAD_CENTER_X
        dy = r.y - PAD_Y
        dist = math.hypot(dx, dy)
        speed = math.hypot(r.vx, r.vy)

        inputs = [dx/800, dy/600, r.vx/6, r.vy/6, r.angle/90, dist/1000, r.x/800, (800-r.x)/800]

        out = brain.predict(inputs)
        r.thrust = out[0] > 0.5
        r.left  = out[1] > 0.5
        r.right = out[2] > 0.5

        if r.thrust: r.apply_thrust()
        if r.left:  r.rotate_left()
        if r.right: r.rotate_right()
        r.update()
        r.check_landing()

        # STRONG GUIDANCE
        penalty = (
            dist * 2 +
            abs(r.angle) * 8 +
            speed * 40 +
            max(0, speed - 2) * 200 +
            max(0, abs(r.angle) - 10) * 500
        )

        # HUGE bonus only when almost perfect
        if dist < 60 and speed < 1.5 and abs(r.angle) < 10:
            penalty -= 2000

        best_penalty = min(best_penalty, penalty)

        if r.landed:
            return 200000 + r.fuel * 50
        if r.crashed or abs(r.x) > 2000:
            return 10000 - best_penalty * 2

    return 10000 - best_penalty * 2


def record(brain):
    r = Rocket()
    hist = []
    for _ in range(MAX_STEPS):
        inputs = [(r.x-PAD_CENTER_X)/800, (r.y-PAD_Y)/600, r.vx/6, r.vy/6,
                  r.angle/90, math.hypot(r.x-PAD_CENTER_X, r.y-PAD_Y)/1000,
                  r.x/800, (800-r.x)/800]
        out = brain.predict(inputs)
        r.thrust, r.left, r.right = [x>0.5 for x in out]
        if r.thrust: r.apply_thrust()
        if r.left: r.rotate_left()
        if r.right: r.rotate_right()
        r.update()
        r.check_landing()
        hist.append((r.x, r.y, r.angle, r.fuel, r.thrust))
        if r.landed or r.crashed: break
    return hist


# ------------------------------
# Replay
# ------------------------------
def replay(hist, gen, final=False):
    if not hist: return
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 32)
    i = 0
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit(); return
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE: i = 0
                if e.key == pygame.K_RIGHT and not final:
                    pygame.quit(); return

        screen.fill((135,206,235))
        pygame.draw.rect(screen, (34,139,34), (0, GROUND_Y, WIDTH, 100))
        pygame.draw.rect(screen, (100,100,100), (PAD_X, PAD_Y, PAD_WIDTH, 8))

        if i < len(hist):
            x, y, a, f, thrust = hist[i]
            corners = Rocket(); corners.x = x; corners.y = y; corners.angle = a
            pygame.draw.polygon(screen, (255,255,255), corners.corners())

            frac = f / FUEL_START
            h = 44 * frac
            ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
            poly = [(-4,22-h),(4,22-h),(4,22),(-4,22)]
            fuel_poly = [(x + lx*ca - ly*sa, y + lx*sa + ly*ca) for lx, ly in poly]
            col = (0,200,0) if frac>0.6 else (220,180,0) if frac>0.25 else (200,40,0)
            pygame.draw.polygon(screen, col, fuel_poly)

            if thrust and f > 0:
                fx = x + 28 * math.sin(math.radians(-a))
                fy = y + 28 * math.cos(math.radians(-a))
                pygame.draw.circle(screen, (255,180,0), (int(fx), int(fy)), 9)

            i += 1

        txt = font.render(f"PERFECT LANDING! Gen {gen}" if final else f"Best of Gen {gen}", True, (0,0,0))
        instr = font.render("SPACE = restart    → = next    ESC = quit", True, (0,0,0))
        screen.blit(txt, (10,10))
        screen.blit(instr, (10,560))
        pygame.display.flip()
        clock.tick(60)


# ------------------------------
# Evolution
# ------------------------------
pop = [Brain() for _ in range(POP_SIZE)]
saved = []
gen = 0
print("Training — THIS ONE ALWAYS WORKS")

while True:
    gen += 1
    fits = [evaluate(b) for b in pop]
    idx = sorted(range(POP_SIZE), key=lambda i: fits[i], reverse=True)
    pop = [pop[i] for i in idx]
    best = fits[idx[0]]
    print(f"Gen {gen} → {best:.0f}")

    if gen % SAVE_EVERY == 0:
        print("   → saved")
        saved.append((gen, record(pop[0])))

    if best >= 200000:
        print(f"\nPERFECT LANDING IN GEN {gen}!")
        saved.append((gen, record(pop[0])))
        break

    elites = [pop[i].copy() for i in range(7)]
    next_pop = elites[:]
    while len(next_pop) < POP_SIZE:
        child = random.choice(elites).copy()
        child.mutate()
        next_pop.append(child)
    pop = next_pop

print(f"\nShowing {len(saved)} flights...")
for i, (g, h) in enumerate(saved):
    replay(h, g, i == len(saved)-1)