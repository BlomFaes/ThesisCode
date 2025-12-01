import pygame
import math
import sys
import random

# ------------------------------------------------------------
# Configuration – simple ON/OFF wind with W key
# ------------------------------------------------------------
WIDTH, HEIGHT = 800, 600
SKY_BLUE      = (135, 206, 235)
GROUND_GREEN  = (34, 139, 34)
PAD_GREY      = (128, 128, 128)
WHITE         = (255, 255, 255)
YELLOW        = (255, 220, 50)

GRAVITY       = 12.0
DT            = 0.016
THRUST_POWER  = 700.0
SIDE_POWER    = 1800.0
WIND_FORCE    = 8.0

# landing tolerance
MAX_LAND_SPEED = 8.0

# wind visualisation
wind_lines = []
particles = []

# ------------------------------------------------------------
# Wind line spawner – only when wind is ON (rightward)
# ------------------------------------------------------------
def spawn_wind_line():
    y = random.randint(50, HEIGHT-100)
    wind_lines.append({
        'pos': pygame.Vector2(-50, y),
        'vel': pygame.Vector2(WIND_FORCE * 10, 0),
        'life': 2.0
    })

# ------------------------------------------------------------
# Thruster particles – small, vertical, under rocket only
# ------------------------------------------------------------
def spawn_main_particles(rocket):
    if rocket.fuel <= 0 or rocket.landed or rocket.crashed: return
    base = pygame.Vector2(0, rocket.size[1]//2).rotate(rocket.angle) + rocket.pos
    width = rocket.size[0] * 0.8
    for _ in range(8):
        offset_x = random.uniform(-width/2, width/2)
        speed = random.uniform(80, 150)
        life = random.uniform(0.2, 0.4)
        particles.append({
            'pos': base + pygame.Vector2(offset_x, 0),
            'vel': pygame.Vector2(0, speed).rotate(rocket.angle),
            'life': life,
            'max_life': life,
            'color': random.choice([YELLOW, WHITE]),
            'size': random.uniform(3, 6)
        })

def spawn_side_particles(rocket, side):
    if rocket.fuel <= 0 or rocket.landed or rocket.crashed: return
    offset = pygame.Vector2(rocket.size[0]//2, 0).rotate(rocket.angle)
    if side == 'left': offset = -offset
    tip = offset + rocket.pos
    for _ in range(3):
        angle = rocket.angle + 90 + random.uniform(-30, 30) if side == 'left' else rocket.angle - 90 + random.uniform(-30, 30)
        speed = random.uniform(60, 100)
        life = random.uniform(0.15, 0.25)
        particles.append({
            'pos': tip.copy(),
            'vel': pygame.Vector2(0, 1).rotate(angle) * speed,
            'life': life,
            'max_life': life,
            'color': WHITE,
            'size': random.uniform(2, 4)
        })

# ------------------------------------------------------------
# Rocket – 2× size, no auto-upright
# ------------------------------------------------------------
class Rocket:
    def __init__(self, x, y, mass=10.0, fuel=100.0):
        self.pos   = pygame.Vector2(x, y)
        self.vel   = pygame.Vector2(0, 0)
        self.angle = 0.0
        self.omega = 0.0
        self.mass  = mass
        self.fuel  = fuel
        self.size  = (12, 80)
        self.landed = False
        self.crashed = False

    def main_thrust(self, dt):
        if self.fuel <= 0 or self.landed or self.crashed: return
        direction = pygame.Vector2(0, -1).rotate(self.angle)
        acc = direction * (THRUST_POWER / self.mass)
        self.vel += acc * dt
        self.fuel -= 1.0 * dt
        if self.fuel < 0: self.fuel = 0
        spawn_main_particles(self)

    def left_thruster(self, dt):
        if self.fuel <= 0 or self.landed or self.crashed: return
        ang_acc = SIDE_POWER / self.mass
        self.omega -= ang_acc * dt
        self.fuel -= 0.5 * dt
        if self.fuel < 0: self.fuel = 0
        spawn_side_particles(self, 'left')

    def right_thruster(self, dt):
        if self.fuel <= 0 or self.landed or self.crashed: return
        ang_acc = SIDE_POWER / self.mass
        self.omega += ang_acc * dt
        self.fuel -= 0.5 * dt
        if self.fuel < 0: self.fuel = 0
        spawn_side_particles(self, 'right')

    def update(self, dt, wind=0.0, pad_top=None, floor_y=None):
        if self.landed or self.crashed:
            self.vel = pygame.Vector2(0, 0)
            self.omega = 0.0
            return

        self.vel.y += GRAVITY * dt
        self.vel.x += (wind / self.mass) * dt
        self.pos   += self.vel * dt
        self.angle += self.omega * dt
        self.omega *= 0.99
        self.angle = (self.angle + 180) % 360 - 180

        half_h = self.size[1] / 2
        bottom_y = self.pos.y + half_h

        if pad_top is not None and bottom_y >= pad_top:
            self.pos.y = pad_top - half_h
            self.vel.y = 0
            return

        if floor_y is not None and bottom_y >= floor_y:
            self.pos.y = floor_y - half_h
            self.vel.y = 0

    def points(self):
        hw, hh = self.size[0] / 2, self.size[1] / 2
        corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        return [pygame.Vector2(p).rotate(self.angle) + self.pos for p in corners]

    def speed(self):          return self.vel.length()
    def location(self):       return self.pos
    def angular_momentum(self):
        I = self.mass * (self.size[1] / 2) ** 2
        return I * math.radians(self.omega)

# ------------------------------------------------------------
# Main loop – **simple W = toggle wind ON/OFF**
# ------------------------------------------------------------
def main(render=True):
    if render:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Rocket Landing – Simple Wind")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 24)

    rocket = Rocket(WIDTH / 2, 50)
    pad    = pygame.Rect(WIDTH / 2 - 100, HEIGHT - 40, 200, 20)
    floor  = pygame.Rect(0, HEIGHT - 20, WIDTH, 20)

    wind_on = False           # simple boolean
    last_wind_toggle = 0
    wind_spawn_timer = 0.0
    landing_message_shown = False

    running = True
    while running:
        if render:
            dt = clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
        else:
            dt = DT

        # ----- input -------------------------------------------------
        if render:
            keys = pygame.key.get_pressed()
            now = pygame.time.get_ticks()

            # **W = toggle wind ON/OFF**
            if keys[pygame.K_w] and now - last_wind_toggle > 300:
                wind_on = not wind_on
                last_wind_toggle = now

            if keys[pygame.K_SPACE]:
                rocket.main_thrust(dt)
            if keys[pygame.K_a]:
                rocket.left_thruster(dt)
            if keys[pygame.K_d]:
                rocket.right_thruster(dt)

        # ----- wind line spawning (only when ON) --------------------
        if wind_on and render:
            wind_spawn_timer -= dt
            if wind_spawn_timer <= 0:
                spawn_wind_line()
                wind_spawn_timer = 0.07

        # ----- physics ------------------------------------------------
        wind = WIND_FORCE if wind_on else 0.0
        rocket.update(dt, wind, pad_top=pad.top, floor_y=floor.top)

        # ----- landing / crash ----------------------------------------
        if not (rocket.landed or rocket.crashed):
            half_h = rocket.size[1] / 2
            bottom_y = rocket.pos.y + half_h

            if pad.left <= rocket.pos.x <= pad.right and bottom_y >= pad.top:
                if abs(rocket.vel.y) <= MAX_LAND_SPEED:
                    rocket.landed = True
                    rocket.vel = pygame.Vector2(0, 0)
                    rocket.omega = 0.0
                    if render and not landing_message_shown:
                        print("SUCCESS – landed!")
                        landing_message_shown = True
                else:
                    rocket.crashed = True
                    rocket.vel = pygame.Vector2(0, 0)
                    rocket.omega = 0.0
                    if render and not landing_message_shown:
                        print("CRASH! – too fast")
                        landing_message_shown = True

            elif bottom_y >= floor.top:
                rocket.crashed = True
                rocket.vel = pygame.Vector2(0, 0)
                rocket.omega = 0.0
                if render and not landing_message_shown:
                    print("CRASH! – missed pad")
                    landing_message_shown = True

        # ----- update particles & wind lines -------------------------
        for p in particles[:]:
            p['pos'] += p['vel'] * dt
            p['life'] -= dt
            if p['life'] <= 0:
                particles.remove(p)
            else:
                p['vel'].y += 200 * dt

        for w in wind_lines[:]:
            w['pos'] += w['vel'] * dt
            w['life'] -= dt
            if w['life'] <= 0 or w['pos'].x > WIDTH + 100:
                wind_lines.remove(w)

        # ----- rendering ---------------------------------------------
        if render:
            screen.fill(SKY_BLUE)
            pygame.draw.rect(screen, GROUND_GREEN, floor)
            pygame.draw.rect(screen, PAD_GREY, pad)

            # wind lines
            if wind_on:
                for w in wind_lines:
                    alpha = int(180 * (w['life'] / 2.0))
                    s = pygame.Surface((30, 2), pygame.SRCALPHA)
                    s.fill((255, 255, 255, alpha))
                    screen.blit(s, (w['pos'].x, w['pos'].y))

            # rocket
            pygame.draw.polygon(screen, WHITE, rocket.points())

            # thruster particles
            for p in particles:
                alpha = int(255 * (p['life'] / p['max_life']))
                size = int(p.get('size', 4))
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                s.fill((*p['color'][:3], alpha))
                screen.blit(s, (p['pos'].x - size, p['pos'].y - size))

            # HUD
            status = "LANDED" if rocket.landed else ("CRASHED" if rocket.crashed else "FLYING")
            info = (f"Fuel: {rocket.fuel:.1f} | "
                    f"Speed: {rocket.speed():.1f} | "
                    f"Angle: {rocket.angle:+.1f} degrees | "
                    f"Wind: {'ON' if wind_on else 'OFF'} | "
                    f"Status: {status}")
            txt = font.render(info, True, (0, 0, 0))
            screen.blit(txt, (10, 10))

            pygame.display.flip()

    if render:
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    main(render=True)