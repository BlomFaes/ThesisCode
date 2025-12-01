import math
import pygame
import random

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

START_X = 100
START_Y = 50
START_VX = 5
START_VY = 1
START_ANGLE = -80

SAVE_EVERY = 10
POP_SIZE = 50
MAX_STEPS = 800
FUEL_START = 350.0

MUTATION_RATE = 0.2
MUTATION_STRENGTH = 1

# ------------------------------
# Brain
# ------------------------------
def sigmoid(x):
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
        self.x = START_X;
        self.y = START_Y;
        self.vx = START_VX;
        self.vy = START_VY

        self.angle = START_ANGLE;
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
            self.fuel -= 0.25
            self.angle -= 0.8

    def rotate_right(self):
        if self.fuel > 0 and not self.landed:
            self.fuel -= 0.25
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
            if speed < 1 and abs(self.angle) < 3 and abs(self.vx) < 0.6:
                self.landed = True
                self.vx = self.vy = 0
                self.thrust = False
                # Critical fix: set y so bottom exactly touches PAD_Y
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
        inputs = [ # all normalized to approximatly -1 to 1
            dx / WIDTH,  # 1. horizontal distance to pad
            (r.y-PAD_Y) / HEIGHT,  # 2. vertical distance to pad
            r.vx / 8,  # 3. horizontal velocity
            r.vy / 8,  # 4. vertical velocity
            r.angle / 90,  # 5. current angle (in degrees)
            dist / 1000,  # 6. straight-line distance to pad
        ]
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
            inputs = [dx/WIDTH, dy/HEIGHT, r.vx/8, r.vy/8, r.angle/90,
                      math.hypot(dx,dy)/1000]
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
# Replay
# ------------------------------
def replay_generation(history, gen, is_final=False):
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

        screen.fill((100,200,250)) # the sky
        pygame.draw.rect(screen, (50,150,50), (0, GROUND_Y, WIDTH, 100)) #the ground
        pygame.draw.rect(screen, (42,42,42), (PAD_X, PAD_Y, PAD_WIDTH, 8)) #the landing pad

        if is_final:
            # FINAL:
            if i < len(history[0]):
                x,y,a,f,thrust,landed,crashed = history[0][i]
                if not crashed:
                    corners = Rocket(); corners.x,corners.y,corners.angle = x,y,a
                    pygame.draw.polygon(screen, (255,255,255), corners.corners())

                    # Fuel bar
                    frac = f/FUEL_START
                    h = (ROCKET_HEIGHT-6) * frac
                    ca,sa = math.cos(math.radians(a)), math.sin(math.radians(a))
                    poly = [(-(ROCKET_WIDTH/2-1),ROCKET_HEIGHT/2-3-h),
                            (ROCKET_WIDTH/2-1,ROCKET_HEIGHT/2-3-h),
                            (ROCKET_WIDTH/2-1,ROCKET_HEIGHT/2-3),
                            (-(ROCKET_WIDTH/2-1),ROCKET_HEIGHT/2-3)]
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
                x,y,a,f,thrust,_,crashed = history[idx][i]
                if crashed: continue
                corners = Rocket(); corners.x,corners.y,corners.angle = x,y,a
                pygame.draw.polygon(screen, (255,255,255), corners.corners())

                frac = f/FUEL_START
                h = (ROCKET_HEIGHT - 6) * frac
                ca,sa = math.cos(math.radians(a)), math.sin(math.radians(a))
                poly = [(-(ROCKET_WIDTH / 2 - 1), ROCKET_HEIGHT / 2 - 3 - h),
                        (ROCKET_WIDTH / 2 - 1, ROCKET_HEIGHT / 2 - 3 - h),
                        (ROCKET_WIDTH / 2 - 1, ROCKET_HEIGHT / 2 - 3),
                        (-(ROCKET_WIDTH / 2 - 1), ROCKET_HEIGHT / 2 - 3)]
                fuel_poly = [(x + lx*ca - ly*sa, y + lx*sa + ly*ca) for lx,ly in poly]
                col = (0,220,0) if frac>0.6 else (220,180,0) if frac>0.25 else (220,50,50)
                pygame.draw.polygon(screen, col, fuel_poly)
                if thrust and f>0:
                    fx = x + 28*math.sin(math.radians(-a))
                    fy = y + 28*math.cos(math.radians(-a))
                    pygame.draw.circle(screen, (255,180,0), (int(fx),int(fy)), 9)

            # Best rocket
            if i < len(history[0]):
                x,y,a,f,thrust,landed,crashed = history[0][i]
                if not crashed:
                    corners = Rocket(); corners.x,corners.y,corners.angle = x,y,a
                    pygame.draw.polygon(screen, (255,255,255), corners.corners())  # pure cyan

                    frac = f/FUEL_START
                    h = (ROCKET_HEIGHT - 6) * frac
                    ca,sa = math.cos(math.radians(a)), math.sin(math.radians(a))
                    poly = [(-(ROCKET_WIDTH / 2 - 1), ROCKET_HEIGHT / 2 - 3 - h),
                            (ROCKET_WIDTH / 2 - 1, ROCKET_HEIGHT / 2 - 3 - h),
                            (ROCKET_WIDTH / 2 - 1, ROCKET_HEIGHT / 2 - 3),
                            (-(ROCKET_WIDTH / 2 - 1), ROCKET_HEIGHT / 2 - 3)]
                    fuel_poly = [(x + lx*ca - ly*sa, y + lx*sa + ly*ca) for lx,ly in poly]
                    col = (0,255,255) if frac>0.6 else (0,220,255) if frac>0.25 else (0,180,220)
                    pygame.draw.polygon(screen, col, fuel_poly)
                    if thrust and f>0:
                        fx = x + 30*math.sin(math.radians(-a))
                        fy = y + 30*math.cos(math.radians(-a))
                        pygame.draw.circle(screen, (255,255,100), (int(fx),int(fy)), 9)

        title = f"Generation {gen}" if not is_final else f"Final Landing - Gen {gen}"
        txt = font.render(title, True, (255,255,255))
        shadow = font.render(title, True, (0,0,0))
        screen.blit(shadow, (12,12)); screen.blit(txt, (10,10))

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

while True:
    gen += 1
    scores = [evaluate(b) for b in population]
    idx = sorted(range(POP_SIZE), key=lambda i: scores[i], reverse=True)
    population = [population[i] for i in idx]
    best = scores[idx[0]]
    print(f"Gen {gen} → Best: {best:.0f}")

    if gen % SAVE_EVERY == 0 or gen == 1:
        saved.append((gen, record_generation(population)))

    if best >= 10000:
        print(f"\nLANDING ACHIEVED - GEN {gen}")
        saved.append((gen, record_generation(population)))
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
for i, (g, h) in enumerate(saved):
    replay_generation(h, g, i == len(saved)-1)
