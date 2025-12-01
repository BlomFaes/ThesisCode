# rocket_evolution_replay.py
# FINAL VERSION — lands perfectly, stays on pad, centers perfectly

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
POP_SIZE = 60
MAX_STEPS = 1200
FUEL_START = 300.0

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
        h = [0]*16
        for i in range(16):
            s = 0
            for j in range(8):
                s += self.w1[i][j] * inp[j]
            h[i] = sigmoid(s)
        o = [0]*3
        for i in range(3):
            s = 0
            for j in range(16):
                s += self.w2[i][j] * h[j]
            o[i] = sigmoid(s)
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
        if self.fuel <= 0: return
        self.fuel -= 1
        a = math.radians(self.angle)
        self.vx += 0.08 * math.sin(a)
        self.vy -= 0.08 * math.cos(a)

    def rotate_left(self):
        if self.fuel > 0:
            self.fuel -= 0.25
            self.angle -= 1.8

    def rotate_right(self):
        if self.fuel > 0:
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
            angle_ok = abs(self.angle) < 9
            speed_ok = speed < 1.5
            if speed_ok and angle_ok:
                self.landed = True
                self.vx = self.vy = 0
                self.thrust = False
                # Snap to make it stay exactly on pad
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
# PERFECT FITNESS — centers + never stuck
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

        inputs = [
            dx/800, dy/600,
            r.vx/6, r.vy/6,
            r.angle/90,
            dist/1000,
            r.x/800, (800-r.x)/800
        ]

        out = brain.predict(inputs)
        r.thrust = out[0] > 0.5
        r.left  = out[1] > 0.5
        r.right = out[2] > 0.5

        if r.thrust: r.apply_thrust()
        if r.left:  r.rotate_left()
        if r.right: r.rotate_right()
        r.update()
        r.check_landing()

        # NEW: very strong center + angle + speed guidance
        penalty = (
            dist * 3 +           # get close
            horiz_dist * 5 +     # get to center!
            abs(r.angle) * 10 +  # be vertical
            speed * 50 +         # be slow
            max(0, speed - 1.5) * 300 +
            max(0, abs(r.angle) - 8) * 800
        )

        # huge reward only when perfect
        if dist < 50 and speed < 1.3 and abs(r.angle) < 8:
            penalty -= 3000

        best = min(best, penalty)

        if r.landed:
            center_bonus = 10000 * (1 - horiz_dist / 40)  # extra for center
            return 300000 + r.fuel * 60 + center_bonus
        if r.crashed or abs(r.x) > 2000:
            return 5000 - best * 3

    return 5000 - best * 3


def record(brain):
    r = Rocket()
    hist = []
    for _ in range(MAX_STEPS):
        inputs = [
            (r.x-PAD_CENTER_X)/800, (r.y-PAD_Y)/600,
            r.vx/6, r.vy/6, r.angle/90,
            math.hypot(r.x-PAD_CENTER_X, r.y-PAD_Y)/1000,
            r.x/800, (800-r.x)/800
        ]
        out = brain.predict(inputs)
        r.thrust, r.left, r.right = [x>0.5 for x in out]
        if r.thrust: r.apply_thrust()
        if r.left: r.rotate_left()
        if r.right: r.rotate_right()
        r.update()
        r.check_landing()

        # Save state — even after landing!
        hist.append((r.x, r.y, r.angle, r.fuel, r.thrust, r.landed))

        if r.crashed: break
        if r.landed and len(hist) > 60: break  # stay on screen for a bit

    return hist


# ------------------------------
# Replay — rocket STAYS after landing
# ------------------------------
def replay(hist, gen, final=False):
    if not hist: return
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 34)
    i = 0
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE: i = 0
                if e.key == pygame.K_RIGHT and not final:
                    pygame.quit()
                    return

        screen.fill((135,206,235))
        pygame.draw.rect(screen, (34,139,34), (0, GROUND_Y, WIDTH, 100))
        pygame.draw.rect(screen, (100,100,100), (PAD_X, PAD_Y, PAD_WIDTH, 8))

        if i < len(hist):
            x, y, a, f, thrust, landed = hist[i]

            # Draw rocket — works even after landing
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

            # Flame only while thrusting
            if thrust and f > 0:
                fx = x + 28 * math.sin(math.radians(-a))
                fy = y + 28 * math.cos(math.radians(-a))
                pygame.draw.circle(screen, (255,180,0), (int(fx), int(fy)), 9)

            i += 1
        else:
            # Stay on last frame forever
            x, y, a, f, _, _ = hist[-1]
            corners = Rocket()
            corners.x, corners.y, corners.angle = x, y, a
            pygame.draw.polygon(screen, (255,255,255), corners.corners())

        title = f"PERFECT LANDING! Gen {gen}" if final else f"Best of Gen {gen}"
        txt1 = font.render(title, True, (0,0,0))
        txt2 = font.render("SPACE = restart    Right Arrow = next    ESC = quit", True, (0,0,0))
        screen.blit(txt1, (10,10))
        screen.blit(txt2, (10,560))

        pygame.display.flip()
        clock.tick(60)


# ------------------------------
# Evolution
# ------------------------------
pop = [Brain() for _ in range(POP_SIZE)]
saved = []
gen = 0
print("Training — centers perfectly, rocket stays visible")

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

    if best >= 300000:
        print(f"\nLANDED PERFECTLY IN CENTER — Gen {gen}!")
        saved.append((gen, record(pop[0])))
        break

    elites = [pop[i].copy() for i in range(8)]
    next_pop = elites[:]
    while len(next_pop) < POP_SIZE:
        child = random.choice(elites).copy()
        child.mutate()
        next_pop.append(child)
    pop = next_pop

print(f"\nShowing {len(saved)} flights...")
for i, (g, h) in enumerate(saved):
    replay(h, g, i == len(saved)-1)

print("Done! Rocket evolution complete.")