import math
import random
import sys
import pygame
from stable_baselines3 import PPO
import numpy as np

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

MODEL_PATH = "output/ppo_mlp_1M"
custom_objects = {
    "learning_rate": 0.0,
    "lr_schedule": lambda _: 0.0,
    "clip_range": lambda _: 0.1,
}
ppo_model = PPO.load(MODEL_PATH, custom_objects=custom_objects)

def get_agent_action(ball, right):
    if ppo_model is not None:
        obs = np.array([
            ball.x / WIDTH * 2 - 1,
            ball.y / HEIGHT * 2 - 1,
            ball.vx / BALL_SPEED_X,
            ball.vy / BALL_SPEED_Y,
            right.center_y / HEIGHT * 2 - 1
        ], dtype=np.float32)
        action, _ = ppo_model.predict(obs, deterministic=True)
        c_min = BORDER + PADDLE_H / 2
        c_max = HEIGHT - BORDER - PADDLE_H / 2
        target_y = (action[0] + 1.0) / 2.0 * (c_max - c_min) + c_min
        return float(target_y)
    else:
        return random_agent_action()

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

        # always toward player (right)
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


# -------------------- AI -------------------- #
def unbeatable_ai(paddle, ball):
    if ball.vx < 0:
        paddle.target_y = ball.center_y
    else:
        paddle.target_y = HEIGHT / 2

def random_agent_action():
    return random.uniform(BORDER, HEIGHT - BORDER)


# -------------------- DRAW -------------------- #
def draw_background(screen):
    screen.fill(BG)

    pygame.draw.rect(screen, FG, (0, 0, WIDTH, BORDER))
    pygame.draw.rect(screen, FG, (0, HEIGHT - BORDER, WIDTH, BORDER))

    # dashed midline
    x = WIDTH // 2
    y = BORDER + 10
    while y < HEIGHT - BORDER:
        pygame.draw.rect(screen, FG, (x - 2, y, 4, DASH_H))
        y += DASH_H + DASH_GAP


# -------------------- MAIN -------------------- #
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pong RL Environment")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 60)

    left = Paddle(LEFT_X, HEIGHT / 2)
    right = Paddle(RIGHT_X, HEIGHT / 2)

    ball = Ball()

    left_score = 0
    right_score = 0

    agent_can_act = False
    first_phase = True  # <-- tracking phase

    running = True
    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # -------- LEFT AI --------
        unbeatable_ai(left, ball)

        # -------- RIGHT (AGENT LOGIC) --------
        if first_phase:
            # perfect alignment (no motion delay)
            right.y = ball.center_y - PADDLE_H / 2
            right.y = clamp(right.y, BORDER, HEIGHT - BORDER - PADDLE_H)
        else:
            if agent_can_act:
                print("Agent is acting")
                right.target_y = get_agent_action(ball, right)
                print("Agent target y: ", right.target_y)
                agent_can_act = False

        # -------- UPDATE --------
        left.update()
        right.update()
        ball.update()

        # -------- COLLISIONS --------
        if ball.rect.colliderect(left.rect):
            ball.vx = abs(ball.vx)

        if ball.rect.colliderect(right.rect):
            ball.vx = -abs(ball.vx)

            if first_phase:
                # after first hit → switch to RL mode
                first_phase = False
                agent_can_act = True
            else:
                agent_can_act = True

        # -------- SCORE --------
        if ball.x < 0:
            right_score += 1
            ball.reset()
            first_phase = True
            agent_can_act = False

        elif ball.x > WIDTH:
            left_score += 1
            ball.reset()
            first_phase = True
            agent_can_act = False

        # -------- DRAW --------
        draw_background(screen)

        pygame.draw.rect(screen, FG, left.rect)
        pygame.draw.rect(screen, FG, right.rect)
        pygame.draw.rect(screen, FG, ball.rect)

        # score
        left_s = font.render(str(left_score), True, FG)
        right_s = font.render(str(right_score), True, FG)

        screen.blit(left_s, (WIDTH // 2 - 120, 20))
        screen.blit(right_s, (WIDTH // 2 + 60, 20))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
