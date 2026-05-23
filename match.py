"""
MLP vs CNN match in pygame.

Both models were trained as the RIGHT paddle against an unbeatable AI.
We put CNN on the right (it sees the world as it trained) and MLP on
the left, but we mirror MLP's observations horizontally so it still
thinks it's playing on the right side.

Run:
    python match.py
    python3 match.py --mlp output/ppo_mlp_elr --cnn output/ppo_cnn_stop --fps 30
    python match.py --swap     # MLP on right, CNN on left
"""

import argparse
import sys
import numpy as np
from collections import deque
import pygame

from stable_baselines3 import PPO
import cv2

from pong import (
    Paddle, Ball, clamp,
    WIDTH, HEIGHT, BORDER, PADDLE_W, PADDLE_H,
    LEFT_X, RIGHT_X, BALL_SIZE,
    BALL_SPEED_X, BALL_SPEED_Y,
    FG, draw_background,
)


C_MIN = BORDER + PADDLE_H / 2
C_MAX = HEIGHT - BORDER - PADDLE_H / 2


# ====================================================================== #
#                       OBS BUILDERS (training-aligned)                  #
# ====================================================================== #
def mlp_obs_as_right(ball, my_paddle, mirror_x: bool):
    """Build MLP's 5-dim obs. If mirror_x=True, we're actually playing
    LEFT but pretend to be RIGHT by mirroring x and vx."""
    if mirror_x:
        bx = WIDTH - ball.x          # mirror position
        bvx = -ball.vx               # mirror velocity
    else:
        bx = ball.x
        bvx = ball.vx
    return np.array([
        bx / WIDTH * 2 - 1,
        ball.y / HEIGHT * 2 - 1,
        bvx / BALL_SPEED_X,
        ball.vy / BALL_SPEED_Y,
        my_paddle.center_y / HEIGHT * 2 - 1,
    ], dtype=np.float32)


# CNN needs the same offscreen rendering as during training.
_obs_surface = None

def _ensure_obs_surface():
    global _obs_surface
    if _obs_surface is None:
        _obs_surface = pygame.Surface((WIDTH, HEIGHT))


def _render_obs_84(left, right, ball, mirror_x: bool = False):
    """Render the scene to (84, 84) uint8 grayscale, identical to
    training. mirror_x flips the rendering horizontally — used if
    CNN is the left player."""
    _ensure_obs_surface()
    draw_background(_obs_surface)
    pygame.draw.rect(_obs_surface, FG, left.rect)
    pygame.draw.rect(_obs_surface, FG, right.rect)
    pygame.draw.rect(_obs_surface, FG, ball.rect)

    surface_array = pygame.surfarray.array3d(_obs_surface)
    frame = np.transpose(surface_array, (1, 0, 2))
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

    if mirror_x:
        img = img[:, ::-1].copy()        # flip horizontally
    return img.astype(np.uint8)


# ====================================================================== #
#                            AGENT ADAPTERS                              #
# ====================================================================== #
class MLPAgent:
    """Plays either side. Internally always thinks it's the right paddle."""
    def __init__(self, model_path: str, side: str):
        assert side in ("left", "right")
        
        # Override the pickled lr_schedule to avoid cloudpickle segmentation faults
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.1,
        }
        self.model = PPO.load(model_path, custom_objects=custom_objects)
        self.side = side                 # which paddle this agent controls
        self.mirror = (side == "left")   # mirror obs if on left

    def act(self, ball, left, right):
        my = left if self.side == "left" else right
        obs = mlp_obs_as_right(ball, my, mirror_x=self.mirror)
        action, _ = self.model.predict(obs, deterministic=True)
        a = float(np.clip(action[0], -1.0, 1.0))
        # y axis is the same in both real and mirrored worlds
        return (a + 1.0) / 2.0 * (C_MAX - C_MIN) + C_MIN


class CNNAgent:
    """Plays either side. Maintains frame stack in *its* reference frame."""
    def __init__(self, model_path: str, side: str):
        assert side in ("left", "right")
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.1,
        }
        self.model = PPO.load(model_path, custom_objects=custom_objects)
        self.side = side
        self.mirror = (side == "left")
        self.frame_stack = deque(maxlen=4)

    def reset_state(self):
        self.frame_stack.clear()

    def record_frame(self, left, right, ball):
        """Called every tick to record the current frame."""
        frame = _render_obs_84(left, right, ball, mirror_x=self.mirror)
        self.frame_stack.append(frame)

    def act(self, left, right, ball):
        if len(self.frame_stack) < 4:
            my = left if self.side == "left" else right
            return my.center_y
            
        obs = np.stack(self.frame_stack, axis=-1)
        action, _ = self.model.predict(obs, deterministic=True)
        a = float(np.clip(action[0], -1.0, 1.0))
        return (a + 1.0) / 2.0 * (C_MAX - C_MIN) + C_MIN


# ====================================================================== #
#                              MAIN LOOP                                 #
# ====================================================================== #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlp", default="output/ppo_mlp_elr.zip")
    parser.add_argument("--cnn", default="output/ppo_cnn_stop.zip")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--swap", action="store_true",
                        help="Put MLP on right, CNN on left (default: opposite)")
    args = parser.parse_args()

    if args.swap:
        left_agent = CNNAgent(args.cnn, side="left")
        right_agent = MLPAgent(args.mlp, side="right")
        left_name, right_name = "CNN", "MLP"
    else:
        left_agent = MLPAgent(args.mlp, side="left")
        right_agent = CNNAgent(args.cnn, side="right")
        left_name, right_name = "MLP", "CNN"

    # ---- pygame setup ----
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Pong — {left_name} vs {right_name}")
    clock = pygame.time.Clock()
    score_font = pygame.font.SysFont("consolas", 60)
    label_font = pygame.font.SysFont("consolas", 22)

    # ---- game state ----
    left = Paddle(LEFT_X, HEIGHT / 2)
    right = Paddle(RIGHT_X, HEIGHT / 2)
    left.target_y = HEIGHT / 2
    right.target_y = HEIGHT / 2
    ball = Ball()

    left_score = 0
    right_score = 0

    # Each side has its own "first hit" auto-align,
    # since neither agent has decided anything yet at the start.
    first_phase_right = True       # ball spawns going right, so right is first receiver
    first_phase_left = True        # left's first decision also needs bootstrap
    right_can_act = False
    left_can_act = False

    def reset_after_score():
        nonlocal first_phase_right, first_phase_left
        nonlocal right_can_act, left_can_act
        ball.reset()
        first_phase_right = True
        first_phase_left = True
        right_can_act = False
        left_can_act = False
        # Clear CNN's history on both agents (if CNN is on either side)
        if isinstance(left_agent, CNNAgent):
            left_agent.reset_state()
        if isinstance(right_agent, CNNAgent):
            right_agent.reset_state()

    running = True
    while running:
        clock.tick(args.fps)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False

        # -------- decision phase --------
        # Right side
        if first_phase_right:
            right.y = ball.center_y - PADDLE_H / 2
            right.y = clamp(right.y, BORDER, HEIGHT - BORDER - PADDLE_H)
            right.target_y = ball.center_y
        elif right_can_act:
            right.target_y = right_agent.act(left, right, ball) \
                if isinstance(right_agent, CNNAgent) \
                else right_agent.act(ball, left, right)
            right_can_act = False

        # Left side
        if first_phase_left:
            left.y = ball.center_y - PADDLE_H / 2
            left.y = clamp(left.y, BORDER, HEIGHT - BORDER - PADDLE_H)
            left.target_y = ball.center_y
        elif left_can_act:
            left.target_y = left_agent.act(left, right, ball) \
                if isinstance(left_agent, CNNAgent) \
                else left_agent.act(ball, left, right)
            left_can_act = False

        # -------- physics --------
        left.update()
        right.update()
        ball.update()

        # -------- collisions --------
        if ball.rect.colliderect(left.rect):
            ball.vx = abs(ball.vx)
            if first_phase_left:
                first_phase_left = False
            left_can_act = True

        if ball.rect.colliderect(right.rect):
            ball.vx = -abs(ball.vx)
            if first_phase_right:
                first_phase_right = False
            right_can_act = True

        # -------- record frames for CNN --------
        if isinstance(left_agent, CNNAgent):
            left_agent.record_frame(left, right, ball)
        if isinstance(right_agent, CNNAgent):
            right_agent.record_frame(left, right, ball)

        # -------- score --------
        if ball.x < 0:
            right_score += 1
            reset_after_score()
        elif ball.x > WIDTH:
            left_score += 1
            reset_after_score()

        # -------- draw --------
        draw_background(screen)
        pygame.draw.rect(screen, FG, left.rect)
        pygame.draw.rect(screen, FG, right.rect)
        pygame.draw.rect(screen, FG, ball.rect)

        # Score
        ls = score_font.render(str(left_score), True, FG)
        rs = score_font.render(str(right_score), True, FG)
        screen.blit(ls, (WIDTH // 2 - 120, 20))
        screen.blit(rs, (WIDTH // 2 + 60, 20))

        # Agent labels
        ll = label_font.render(left_name, True, (160, 160, 160))
        rl = label_font.render(right_name, True, (160, 160, 160))
        screen.blit(ll, (20, HEIGHT - 30))
        screen.blit(rl, (WIDTH - rl.get_width() - 20, HEIGHT - 30))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
