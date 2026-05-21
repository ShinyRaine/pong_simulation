import math
import random
import sys
import pygame
from stable_baselines3 import PPO
import numpy as np
import cv2
from collections import deque

WIDTH = 1000
HEIGHT = 600
FPS = 60

BG = (0, 0, 0)
FG = (245, 245, 245)

BORDER = 8

PADDLE_W = 12
PADDLE_H = 80
PADDLE_MARGIN = 18
PADDLE_SPEED = 6

BALL_SIZE = 12
BALL_SPEED_X = 6
BALL_SPEED_Y = 4

LEFT_X = PADDLE_MARGIN
RIGHT_X = WIDTH - PADDLE_MARGIN - PADDLE_W

DASH_H = 14
DASH_GAP = 10

C_MIN = BORDER + PADDLE_H / 2
C_MAX = HEIGHT - BORDER - PADDLE_H / 2

print("Loading MLP Model...")
try:
    mlp_model = PPO.load("output/ppo_baseline")
except Exception as e:
    mlp_model = None
    print("Warning: MLP Model not found.")

print("Loading CNN Model...")
try:
    cnn_model = PPO.load("output/ppo_cnn_dis")
except Exception as e:
    cnn_model = None
    print("Warning: CNN Model not found.")

frame_stack = deque(maxlen=4)

def process_screen(screen):
    surface_array = pygame.surfarray.array3d(screen)
    frame = np.transpose(surface_array, (1, 0, 2))
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized

def get_cnn_action(screen, paddle):
    if cnn_model is None:
        return paddle.center_y

    current_frame = process_screen(screen)
    frame_stack.append(current_frame)

    if len(frame_stack) < 4:
        return paddle.center_y

    obs = np.stack(frame_stack, axis=-1)

    action, _ = cnn_model.predict(obs, deterministic=True)
    
    # Extract integer action
    action_val = int(action.item() if isinstance(action, np.ndarray) else action)

    if action_val == 1:  # UP
        return paddle.center_y - HEIGHT
    elif action_val == 2:  # DOWN
        return paddle.center_y + HEIGHT
    else:  # 0 or STAY
        return paddle.center_y

def get_mlp_action(ball, paddle):
    if mlp_model is None:
        return paddle.center_y
    
    # MLP model was trained on the right side.
    # To use it on the left side, we must flip the X coordinates and X velocities.
    obs = np.array([
        -(ball.x / WIDTH * 2 - 1),
        ball.y / HEIGHT * 2 - 1,
        -(ball.vx / BALL_SPEED_X),
        ball.vy / BALL_SPEED_Y,
        paddle.center_y / HEIGHT * 2 - 1
    ], dtype=np.float32)
    
    action, _ = mlp_model.predict(obs, deterministic=True)
    c_min = BORDER + PADDLE_H / 2
    c_max = HEIGHT - BORDER - PADDLE_H / 2
    target_y = (action[0] + 1.0) / 2.0 * (c_max - c_min) + c_min
    return float(target_y)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# -------------------- OBJECTS -------------------- #
class Paddle:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.target_y = y

    @property
    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), PADDLE_W, PADDLE_H)

    @property
    def center_y(self):
        return self.y + PADDLE_H / 2

    def update(self):
        dy = self.target_y - self.center_y
        
        if abs(dy) < PADDLE_SPEED:
            self.y += dy
        else:
            self.y += PADDLE_SPEED if dy > 0 else -PADDLE_SPEED

        self.y = clamp(self.y, BORDER, HEIGHT - BORDER - PADDLE_H)


class Ball:
    def __init__(self):
        self.reset()

    @property
    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), BALL_SIZE, BALL_SIZE)

    @property
    def center_y(self):
        return self.y + BALL_SIZE / 2

    def reset(self):
        self.x = WIDTH / 2
        self.y = random.uniform(HEIGHT * 0.2, HEIGHT * 0.8)

        self.vx = abs(BALL_SPEED_X)
        vy = random.uniform(-BALL_SPEED_Y, BALL_SPEED_Y)
        if abs(vy) < 1.5:
            vy = 1.5 if vy >= 0 else -1.5
        self.vy = vy

    def update(self):
        self.x += self.vx
        self.y += self.vy

        if self.y <= BORDER or self.y >= HEIGHT - BORDER:
            self.vy *= -1

# -------------------- DRAW -------------------- #
def draw_background(screen):
    screen.fill(BG)
    pygame.draw.rect(screen, FG, (0, 0, WIDTH, BORDER))
    pygame.draw.rect(screen, FG, (0, HEIGHT - BORDER, WIDTH, BORDER))
    x = WIDTH // 2
    y = BORDER + 10
    while y < HEIGHT - BORDER:
        pygame.draw.rect(screen, FG, (x - 2, y, 4, DASH_H))
        y += DASH_H + DASH_GAP

# -------------------- MAIN -------------------- #
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pong: CNN Agent (Right) vs MLP Agent (Left)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 60)

    left = Paddle(LEFT_X, HEIGHT / 2)
    right = Paddle(RIGHT_X, HEIGHT / 2)
    ball = Ball()

    left_score = 0
    right_score = 0
    
    left_first_phase = True
    right_first_phase = True
    mlp_can_act = False

    running = True
    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 1. Render current state so CNN can process it
        draw_background(screen)
        pygame.draw.rect(screen, FG, left.rect)
        pygame.draw.rect(screen, FG, right.rect)
        pygame.draw.rect(screen, FG, ball.rect)

        # 2. Get actions based on current state
        # --- LEFT PADDLE (MLP) ---
        if left_first_phase:
            left.y = ball.center_y - PADDLE_H / 2
            left.y = clamp(left.y, BORDER, HEIGHT - BORDER - PADDLE_H)
            left.target_y = ball.center_y
        else:
            if mlp_can_act:
                left.target_y = get_mlp_action(ball, left)
                mlp_can_act = False

        # --- RIGHT PADDLE (CNN) ---
        if right_first_phase:
            right.y = ball.center_y - PADDLE_H / 2
            right.y = clamp(right.y, BORDER, HEIGHT - BORDER - PADDLE_H)
            right.target_y = ball.center_y
        else:
            right.target_y = get_cnn_action(screen, right)

        # 3. Update physics
        left.update()
        right.update()
        ball.update()

        # 4. Collisions
        if ball.rect.colliderect(left.rect):
            ball.vx = abs(ball.vx)
            if left_first_phase:
                left_first_phase = False
            mlp_can_act = True

        if ball.rect.colliderect(right.rect):
            ball.vx = -abs(ball.vx)
            if right_first_phase:
                right_first_phase = False

        # 5. Score handling
        if ball.x < 0:
            right_score += 1
            ball.reset()
            left_first_phase = True
            right_first_phase = True
            mlp_can_act = False
            frame_stack.clear()

        elif ball.x > WIDTH:
            left_score += 1
            ball.reset()
            left_first_phase = True
            right_first_phase = True
            mlp_can_act = False
            frame_stack.clear()

        # 6. Draw score
        left_s = font.render(str(left_score), True, FG)
        right_s = font.render(str(right_score), True, FG)
        screen.blit(left_s, (WIDTH // 2 - 120, 20))
        screen.blit(right_s, (WIDTH // 2 + 60, 20))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
