# rocket_evolution_replay.py
import numpy as np
import math
import pygame
from copy import deepcopy

# ------------------------------
# Rocket
# ------------------------------
class Rocket:
    def __init__(self, x, y, vx=0, vy=0, angle=0, fuel=300.0):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.angle = angle
        self.fuel = self.max_fuel = fuel
        self.main_thruster = self.left_thruster = self.right_thruster = False
        self.crashed = self.landed = False

        self.gravity = 0.05
        self.thrust_power = 0.08
        self.rotation_speed = 1.5

    def apply_main_thrust(self):
        if self.fuel <= 0: return
        self.fuel = max(0.0, self.fuel - 1.0)
        a = math.radians(self.angle)
        self.vx += self.thrust_power * math.sin(a)
        self.vy -= self.thrust_power * math.cos(a)

    def rotate_left(self):   self.angle -= self.rotation_speed if self.fuel > 0 else 0; self.fuel -= 0.25
    def rotate_right(self):  self.angle += self.rotation_speed if self.fuel > 0 else 0; self.fuel -= 0.25

    def update(self):
        if self.crashed or self.landed: return
        self.vy += self.gravity
        self.x += self.vx
        self.y += self.vy

    def get_bottom_y(self):
        a = math.radians(self.angle)
        c, s = math.cos(a), math.sin(a)
        return max((cx*s + cy*c + self.y) for cx, cy in [(-5,-25),(5,-25),(5,25),(-5,25)])

    def check_landing(self, pad_x, pad_y, pad_w, pad_h, ground_y):
        if self.crashed or self.landed: return
        bottom = self.get_bottom_y()
        on_pad = pad_x <= self.x <= pad_x + pad_w
        if on_pad and bottom >= pad_y:
            if math.hypot(self.vx, self.vy) < 1.0 and abs(self.angle) < 5:
                self.landed = True
                self.vx = self.vy = 0
                self.main_thruster = False
                self.y -= (bottom - pad_y)
            else:
                self.crashed = True
        elif bottom >= ground_y:
            self.crashed = True

    def get_corners(self):
        a = math.radians(self.angle)
        c, s = math.cos(a), math.sin(a)
        return [(self.x + (x*c - y*s), self.y + (x*s + y*c)) for x, y in [(-5,-25),(5,-25),(5,25),(-5,25)]]


# ------------------------------
# Environment
# ------------------------------
class RocketEnv:
    def __init__(self, start_x, start_y, start_vx, start_vy, start_angle, fuel=300.0):
        self.w, self.h = 800, 600
        self.ground_y = self.h - 80
        self.pad_x = 360
        self.pad_y = self.ground_y - 8
        self.pad_cx = 400
        self.rocket = Rocket(start_x, start_y, start_vx, start_vy, start_angle, fuel)

    def step(self, actions):
        r = self.rocket
        if r.crashed or r.landed: return
        r.main_thruster, r.left_thruster, r.right_thruster = actions['thrust'], actions['left'], actions['right']
        if r.main_thruster: r.apply_main_thrust()
        if r.left_thruster: r.rotate_left()
        if r.right_thruster: r.rotate_right()
        r.update()
        r.check_landing(self.pad_x, self.pad_y, 80, 8, self.ground_y)


# ------------------------------
# Neural Network
# ------------------------------
def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class SimpleNN:
    def __init__(self):
        self.w1 = np.random.randn(12, 8) * 0.5
        self.w2 = np.random.randn(3, 12) * 0.5
        self.b1 = np.random.randn(12, 1) * 0.5
        self.b2 = np.random.randn(3, 1) * 0.5

    def forward(self, x):
        h = sigmoid(self.w1 @ x + self.b1)
        o = sigmoid(self.w2 @ h + self.b2)
        return o.flatten()

    def mutate(self, rate=0.15, scale=0.3):
        for arr in [self.w1, self.w2, self.b1, self.b2]:
            mask = np.random.rand(*arr.shape) < rate
            arr += mask * np.random.randn(*arr.shape) * scale

    def copy(self): return deepcopy(self)


# ------------------------------
# NEW SMOOTH FITNESS (this is the magic)
# ------------------------------
def evaluate(nn, start_x, start_y, start_vx, start_vy, start_angle):
    env = RocketEnv(start_x, start_y, start_vx, start_vy, start_angle)
    r = env.rocket
    best_penalty = 1e9

    for _ in range(1200):
        dx = r.x - env.pad_cx
        dy = r.y - env.pad_y
        dist = math.hypot(dx, dy)

        inputs = np.array([
            dx/800, dy/600,
            r.vx/6, r.vy/6,
            r.angle/90,
            dist/1000,
            r.x/800, (800-r.x)/800
        ]).reshape(-1,1)

        out = nn.forward(inputs)
        actions = {'thrust': out[0]>0.5, 'left': out[1]>0.5, 'right': out[2]>0.5}
        env.step(actions)

        speed = math.hypot(r.vx, r.vy)
        speed_penalty = speed * 20 if dist < 120 else 0
        angle_penalty = abs(r.angle) * 4
        dist_penalty = dist

        bonus = 300 if dist < 80 and speed < 1.5 else 0
        penalty = dist_penalty + speed_penalty + angle_penalty - bonus
        best_penalty = min(best_penalty, penalty)

        if r.landed:
            return 100000 + r.fuel * 30
        if r.crashed or abs(r.x) > 2000 or r.y > 1000:
            return -best_penalty

    return -best_penalty


def record_flight(nn, sx, sy, svx, svy, sa):
    env = RocketEnv(sx, sy, svx, svy, sa)
    r = env.rocket
    history = []
    for _ in range(1200):
        inputs = np.array([
            (r.x-env.pad_cx)/800, (r.y-env.pad_y)/600,
            r.vx/6, r.vy/6, r.angle/90,
            math.hypot(r.x-env.pad_cx, r.y-env.pad_y)/1000,
            r.x/800, (800-r.x)/800
        ]).reshape(-1,1)
        out = nn.forward(inputs)
        actions = {'thrust': out[0]>0.5, 'left': out[1]>0.5, 'right': out[2]>0.5}
        env.step(actions)
        history.append((r.x, r.y, r.angle, r.vx, r.vy, r.fuel, r.main_thruster))
        if r.landed or r.crashed: break
    return history


# ------------------------------
# Clean Replay
# ------------------------------
def play_replay(hist, gen, final=False):
    if not hist: return "done"
    pygame.init()
    screen = pygame.display.set_mode((800,600))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 28)
    i = 0
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit(); return "done"
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE: i = 0
                if e.key == pygame.K_RIGHT and not final:
                    pygame.quit(); return "next"

        screen.fill((135,206,235))
        pygame.draw.rect(screen, (34,139,34), (0,520,800,80))
        pygame.draw.rect(screen, (100,100,100), (360,512,80,8))

        if i < len(hist):
            x,y,a,_,_,f,thrust = hist[i]
            corners = [(x + px*math.cos(math.radians(a)) - py*math.sin(math.radians(a)),
                        y + px*math.sin(math.radians(a)) + py*math.cos(math.radians(a)))
                       for px,py in [(-5,-25),(5,-25),(5,25),(-5,25)]]
            pygame.draw.polygon(screen, (255,255,255), corners)

            # fuel
            frac = f/300
            h = 44*frac
            fuel_poly = [(x + lx*math.cos(math.radians(a)) - ly*math.sin(math.radians(a)),
                          y + lx*math.sin(math.radians(a)) + ly*math.cos(math.radians(a)))
                         for lx,ly in [(-4,22-h),(4,22-h),(4,22),(-4,22)]]
            col = (0,200,0) if frac>0.6 else (220,180,0) if frac>0.25 else (200,40,0)
            pygame.draw.polygon(screen, col, fuel_poly)

            if thrust and f > 0:
                fx = x + 25*math.sin(math.radians(-a))
                fy = y + 25*math.cos(math.radians(-a))
                pygame.draw.circle(screen, (255,180,0), (int(fx),int(fy)), 8)

            i += 1

        txt = font.render("SUCCESS! Gen "+str(gen) if final else "Best of Gen "+str(gen), True, (0,0,0))
        instr = font.render("SPACE=restart  →=next  ESC=quit", True, (0,0,0))
        screen.blit(txt, (10,10))
        screen.blit(instr, (10,560))
        pygame.display.flip()
        clock.tick(60)


# ------------------------------
# Main
# ------------------------------
def run_evolution():
    # YOUR STARTING CONDITION
    START_X, START_Y = 200, 100
    START_VX, START_VY = 4.0, -1.0
    START_ANGLE = 45

    SAVE_EVERY = 10
    POP_SIZE = 60
    ELITISM = 6

    population = [SimpleNN() for _ in range(POP_SIZE)]
    saves = []
    gen = 0

    print(f"Starting from X={START_X} Y={START_Y} VX={START_VX} VY={START_VY} Angle={START_ANGLE}°")
    print("Training... (new smooth fitness)")

    while True:
        gen += 1
        print(f"Gen {gen}", end="")

        fits = [evaluate(nn, START_X, START_Y, START_VX, START_VY, START_ANGLE) for nn in population]
        idx = np.argsort(fits)[::-1]
        population = [population[i] for i in idx]
        best = fits[idx[0]]
        print(f" → {best:.0f}")

        if gen % SAVE_EVERY == 0:
            h = record_flight(population[0], START_X, START_Y, START_VX, START_VY, START_ANGLE)
            saves.append((gen, h))

        if best >= 100000:
            print(f"\nLANDED IN GENERATION {gen}!")
            h = record_flight(population[0], START_X, START_Y, START_VX, START_VY, START_ANGLE)
            saves.append((gen, h))
            break

        next_gen = [p.copy() for p in population[:ELITISM]]
        while len(next_gen) < POP_SIZE:
            p = population[np.random.randint(ELITISM)].copy()
            p.mutate()
            next_gen.append(p)
        population = next_gen

    # Show all saved flights
    for i, (g, h) in enumerate(saves):
        play_replay(h, g, i == len(saves)-1)

if __name__ == "__main__":
    run_evolution()