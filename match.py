"""
MLP vs CNN match in pygame.

Both models were trained as the RIGHT paddle against an unbeatable AI.
We put CNN on the right (it sees the world as it trained) and MLP on
the left, but we mirror MLP's observations horizontally so it still
thinks it's playing on the right side.

Run:
    python match.py
    python match.py --mlp my_mlp.zip --cnn my_cnn.zip --fps 30
    python match.py --swap     # MLP on right, CNN on left
"""

import argparse
import sys
import numpy as np
import pygame
import cv2

from stable_baselines3 import PPO

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
        self.model = PPO.load(model_path)
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
    """Plays either side. Maintains last_left_hit_obs in *its* reference frame.

    Important: 'left hit' for this agent means the OPPONENT's paddle hit, i.e.
    the paddle on the LEFT side of the agent's view. If the agent plays right
    naturally, that's the actual left paddle. If the agent plays left, we
    mirror, and 'left hit' becomes the actual right paddle hit.
    """
    def __init__(self, model_path: str, side: str):
        assert side in ("left", "right")
        self.model = PPO.load(model_path)
        self.side = side
        self.mirror = (side == "left")
        self.last_opp_hit_obs = None     # opponent-side hit (from this agent's view)

    def reset_state(self):
        self.last_opp_hit_obs = None

    def on_opponent_hit(self, left, right, ball):
        """Called when the OPPONENT (this agent's opponent) just hit the ball.
        For right-side CNN this means left paddle hit; for left-side CNN it
        means right paddle hit."""
        self.last_opp_hit_obs = _render_obs_84(left, right, ball, mirror_x=self.mirror)

    def act(self, left, right, ball):
        current = _render_obs_84(left, right, ball, mirror_x=self.mirror)
        old = self.last_opp_hit_obs if self.last_opp_hit_obs is not None else current
        obs = np.stack([old, current], axis=0)            # (2, 84, 84)
        action, _ = self.model.predict(obs, deterministic=True)
        a = float(np.clip(action[0], -1.0, 1.0))
        return (a + 1.0) / 2.0 * (C_MAX - C_MIN) + C_MIN


# ====================================================================== #
#                              MAIN LOOP                                 #
# ====================================================================== #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlp", default="ppo_pong_simple.zip")
    parser.add_argument("--cnn", default="ppo_pong_cnn_pygame.zip")
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
            # CNN on right side sees this as an "opponent hit" → capture for its channel 0
            if isinstance(right_agent, CNNAgent):
                right_agent.on_opponent_hit(left, right, ball)
            if first_phase_left:
                first_phase_left = False
            left_can_act = True

        if ball.rect.colliderect(right.rect):
            ball.vx = -abs(ball.vx)
            # CNN on left side sees this as its "opponent hit"
            if isinstance(left_agent, CNNAgent):
                left_agent.on_opponent_hit(left, right, ball)
            if first_phase_right:
                first_phase_right = False
            right_can_act = True

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
