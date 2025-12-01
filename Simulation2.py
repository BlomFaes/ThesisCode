import pygame
import math
import sys


class Rocket:
    def __init__(self, x, y, vx, vy, angle, width=10, height=50):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.vx = vx
        self.vy = vy
        self.angle = angle  # in degrees, 0 = upright

        # Physics constants
        self.gravity = 0.05
        self.thrust_power = 0.07  # Much weaker thrust - need to hold it
        self.rotation_speed = 0.3

        # State
        self.main_thruster = False
        self.left_thruster = False
        self.right_thruster = False
        self.crashed = False
        self.landed = False
        self.fuel = 125

    def apply_main_thrust(self):
        if self.fuel <= 0:
            return
        """Apply thrust in the direction the rocket is pointing"""
        angle_rad = math.radians(self.angle)
        # Thrust pushes in the direction of the rocket's orientation
        self.vx += self.thrust_power * math.sin(angle_rad)
        self.vy -= self.thrust_power * math.cos(angle_rad)
        self.fuel -= 1

    def rotate_left(self):
        """Rotate counter-clockwise"""
        self.angle -= self.rotation_speed

    def rotate_right(self):
        """Rotate clockwise"""
        self.angle += self.rotation_speed

    def update(self):
        """Update physics"""
        if self.crashed or self.landed:
            return

        # Apply gravity
        self.vy += self.gravity

        # Update position
        self.x += self.vx
        self.y += self.vy

    def get_bottom_y(self):
        """Get the Y coordinate of the bottom of the rocket"""
        # Calculate all corners
        angle_rad = math.radians(self.angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        hw = self.width / 2
        hh = self.height / 2

        # Get all four corners
        corners = [
            (-hw, -hh),  # top-left
            (hw, -hh),  # top-right
            (hw, hh),  # bottom-right
            (-hw, hh)  # bottom-left
        ]

        # Rotate corners and find lowest Y
        max_y = float('-inf')
        for cx, cy in corners:
            ry = cx * sin_a + cy * cos_a + self.y
            max_y = max(max_y, ry)

        return max_y

    def check_landing(self, pad_x, pad_y, pad_width, pad_height, ground_y):
        """Check if landed successfully or crashed"""
        if self.crashed or self.landed:
            return

        # Get the lowest point of the rocket
        rocket_bottom = self.get_bottom_y()

        # Check if rocket center is on landing pad
        rocket_center_x = self.x
        on_pad = (pad_x <= rocket_center_x <= pad_x + pad_width)

        # Landing pad top surface is at pad_y
        if on_pad and rocket_bottom >= pad_y:
            # On pad - check landing criteria
            speed = math.sqrt(self.vx ** 2 + self.vy ** 2)
            angle_ok = abs(self.angle) < 6  # Must be nearly upright

            if speed < 1 and angle_ok:
                # Successful landing!
                self.landed = True
                self.vy = 0
                self.vx = 0
                # Lock position on top of landing pad
                correction = rocket_bottom - pad_y
                self.y -= correction
            else:
                # Too fast or wrong angle on pad = crash
                self.crashed = True
        elif rocket_bottom >= ground_y:
            # Landed on grass = automatic crash
            self.crashed = True

    def get_corners(self):
        """Get the four corners of the rotated rectangle"""
        angle_rad = math.radians(self.angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # Half dimensions
        hw = self.width / 2
        hh = self.height / 2

        # Corner offsets (unrotated)
        corners = [
            (-hw, -hh),  # top-left
            (hw, -hh),  # top-right
            (hw, hh),  # bottom-right
            (-hw, hh)  # bottom-left
        ]

        # Rotate and translate
        rotated = []
        for cx, cy in corners:
            rx = cx * cos_a - cy * sin_a + self.x
            ry = cx * sin_a + cy * cos_a + self.y
            rotated.append((rx, ry))

        return rotated


class Environment:
    def __init__(self, width=800, height=600, headless=False):
        self.width = width
        self.height = height
        self.headless = headless

        # Ground and landing pad
        self.ground_y = height - 80
        self.pad_width = 80
        self.pad_height = 8
        self.pad_x = (width - self.pad_width) // 2
        self.pad_y = self.ground_y - self.pad_height

        # Create rocket
        self.rocket = Rocket(width // 2, height - 400, 0, 0, 0)
        #self.rocket = Rocket(width // 10, 50, 4, 0, -40)

        # Pygame setup
        if not headless:
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Rocket Landing Simulation")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)

    def handle_input(self):
        """Handle keyboard input"""
        keys = pygame.key.get_pressed()

        # Only allow input if not landed or crashed
        if not self.rocket.landed and not self.rocket.crashed:
            self.rocket.main_thruster = keys[pygame.K_SPACE]
            self.rocket.left_thruster = keys[pygame.K_a]
            self.rocket.right_thruster = keys[pygame.K_d]

            if self.rocket.main_thruster:
                self.rocket.apply_main_thrust()
            if self.rocket.left_thruster:
                self.rocket.rotate_left()
            if self.rocket.right_thruster:
                self.rocket.rotate_right()
        else:
            # Reset thruster states when landed/crashed
            self.rocket.main_thruster = False
            self.rocket.left_thruster = False
            self.rocket.right_thruster = False

    def step(self, actions=None):
        """
        Update simulation one step
        """
        if actions:
            self.rocket.main_thruster = actions.get('thrust', False)
            self.rocket.left_thruster = actions.get('left', False)
            self.rocket.right_thruster = actions.get('right', False)

            if self.rocket.main_thruster:
                self.rocket.apply_main_thrust()
            if self.rocket.left_thruster:
                self.rocket.rotate_left()
            if self.rocket.right_thruster:
                self.rocket.rotate_right()

        self.rocket.update()
        self.rocket.check_landing(self.pad_x, self.pad_y, self.pad_width,
                                  self.pad_height, self.ground_y)

    def render(self):
        """Draw everything"""
        if self.headless:
            return

        # Background
        self.screen.fill((135, 206, 235))

        # Ground
        pygame.draw.rect(self.screen, (34, 139, 34),
                         (0, self.ground_y, self.width, self.height - self.ground_y))

        # Landing pad
        pygame.draw.rect(self.screen, (128, 128, 128),
                         (self.pad_x, self.pad_y, self.pad_width, self.pad_height))

        # Rocket
        corners = self.rocket.get_corners()
        color = (255, 255, 255)  # White
        pygame.draw.polygon(self.screen, color, corners)

        # Thrusters
        if self.rocket.main_thruster and not self.rocket.crashed:
            # Main thruster at bottom of rocket (always stays at bottom)
            angle_rad = math.radians(-self.rocket.angle)  # Inverted angle for visuals
            # Calculate bottom position: rocket center + half height in rocket's direction
            offset_x = (self.rocket.height / 2) * math.sin(angle_rad)
            offset_y = (self.rocket.height / 2) * math.cos(angle_rad)
            thruster_x = self.rocket.x + offset_x
            thruster_y = self.rocket.y + offset_y
            pygame.draw.circle(self.screen, (255, 255, 0),
                               (int(thruster_x), int(thruster_y)), 6)

        if self.rocket.left_thruster and not self.rocket.crashed:
            # right thruster
            angle_rad = math.radians(-self.rocket.angle)

            offset_x = -(self.rocket.height / 2) * math.sin(angle_rad) + (self.rocket.width / 2) * math.cos(angle_rad)
            offset_y = -(self.rocket.height / 2) * math.cos(angle_rad) + (self.rocket.width / 2) * math.sin(angle_rad)
            thruster_x = self.rocket.x + offset_x
            thruster_y = self.rocket.y + offset_y
            pygame.draw.circle(self.screen, (0, 191, 255),
                               (int(thruster_x), int(thruster_y)), 4)

        if self.rocket.right_thruster and not self.rocket.crashed:
            # left thruster
            angle_rad = math.radians(-self.rocket.angle)
            # Top of rocket + perpendicular offset to left
            offset_x = -(self.rocket.height / 2) * math.sin(angle_rad) - (self.rocket.width / 2) * math.cos(angle_rad)
            offset_y = -(self.rocket.height / 2) * math.cos(angle_rad) - (self.rocket.width / 2) * math.sin(angle_rad)
            thruster_x = self.rocket.x + offset_x
            thruster_y = self.rocket.y + offset_y
            pygame.draw.circle(self.screen, (0, 191, 255),
                               (int(thruster_x), int(thruster_y)), 4)

        # Status text
        if self.rocket.landed:
            text = self.font.render("LANDED!", True, (0, 255, 0))
            self.screen.blit(text, (self.width // 2 - 70, 50))
        elif self.rocket.crashed:
            text = self.font.render("CRASHED!", True, (255, 0, 0))
            self.screen.blit(text, (self.width // 2 - 80, 50))

        # Info text
        info_font = pygame.font.Font(None, 24)
        speed = math.sqrt(self.rocket.vx ** 2 + self.rocket.vy ** 2)
        info_text = info_font.render(f"Speed: {speed:.1f} | Angle: {self.rocket.angle:.0f}°", True, (0, 0, 0))
        self.screen.blit(info_text, (10, 10))
        fuel_text = self.font.render(f"Fuel: {self.rocket.fuel:.0f}", True, (0, 0, 0))
        self.screen.blit(fuel_text, (10, 40))

        pygame.display.flip()


if __name__ == "__main__":
    # Create the environment (with visuals)
    env = Environment()

    # Main game loop
    running = True
    while running:
        # Check for quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Handle player input and update physics
        env.handle_input()
        env.step()

        # Draw everything
        env.render()

        # Run at 60 FPS
        env.clock.tick(60)

    pygame.quit()
    sys.exit()