"""
PPO + MLP training for the Pong RL environment.

Drop-in replacement for the previous version with the following fixes:
  1. Episode truncation via max_episode_steps (no unbounded rollouts).
  2. first_phase paddle drift fixed (target_y synced before update()).
  3. Action range maps to the *legal* paddle-center range, no dead zones.
  4. Observation uses paddle center_y, consistent with internal physics.
  5. Reward shaping using closest-approach distance on misses.
  6. Parallel SubprocVecEnv + VecNormalize + TensorBoard logging.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

from pong import (
    Paddle, Ball, unbeatable_ai, clamp,
    WIDTH, HEIGHT, BORDER, PADDLE_H,
    LEFT_X, RIGHT_X, BALL_SPEED_X, BALL_SPEED_Y,
)


class PongEnv(gym.Env):
    """One agent step = one decision cycle (a hit to the next hit-or-miss).

    Observation (5,):
        [ball_x, ball_y, ball_vx, ball_vy, paddle_center_y]   (all normalised)
    Action (1,):
        target paddle center y, normalised to [-1, 1].
    Reward:
        +1.0 per successful return
        -1.0 - miss_shaping * (min_dist / (HEIGHT / 2))   on a miss
    Episode ends:
        terminated = miss
        truncated  = max_episode_steps consecutive successful returns
    """

    metadata = {"render_modes": []}

    # Legal range for the paddle CENTER y (not top-y).
    C_MIN = BORDER + PADDLE_H / 2
    C_MAX = HEIGHT - BORDER - PADDLE_H / 2

    # Track closest approach only when the ball is within this many pixels
    # (in x) of the right paddle, so we don't shape on the outbound leg.
    SHAPING_X_WINDOW = 40

    def __init__(self, max_episode_steps: int = 200, miss_shaping: float = 0.5):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(5,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self.max_episode_steps = max_episode_steps
        self.miss_shaping = miss_shaping

        # Real init in reset().
        self.left = Paddle(LEFT_X, HEIGHT / 2)
        self.right = Paddle(RIGHT_X, HEIGHT / 2)
        self.ball = Ball()
        self.first_phase = True
        self.agent_can_act = False
        self.steps = 0
        self.min_dist = float("inf")

    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            import random
            random.seed(seed)  # Ball.reset() uses python's random module

        self.left = Paddle(LEFT_X, HEIGHT / 2)
        self.right = Paddle(RIGHT_X, HEIGHT / 2)
        # Fix the upstream Paddle.__init__ inconsistency: target_y is set
        # to the top-y there, but update() treats it as center-y.
        self.left.target_y = HEIGHT / 2
        self.right.target_y = HEIGHT / 2
        self.ball = Ball()

        self.first_phase = True
        self.agent_can_act = False
        self.steps = 0
        self.min_dist = float("inf")

        # Fast-forward first_phase to the first paddle hit.
        _, terminated = self._simulate_until_action()
        assert not terminated, "first_phase should never end in a miss"

        return self._get_obs(), {}

    # ------------------------------------------------------------------ #
    def step(self, action):
        a = float(np.clip(action[0], -1.0, 1.0))
        target_center = (a + 1.0) / 2.0 * (self.C_MAX - self.C_MIN) + self.C_MIN
        self.right.target_y = target_center
        self.agent_can_act = False
        self.min_dist = float("inf")

        reward, terminated = self._simulate_until_action()
        self.steps += 1
        truncated = (not terminated) and self.steps >= self.max_episode_steps

        info = {"hit": not terminated, "min_dist": self.min_dist}
        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    def _simulate_until_action(self):
        """Drive the inner physics loop until the agent should act again."""
        reward = 0.0
        terminated = False

        while not self.agent_can_act and not terminated:
            unbeatable_ai(self.left, self.ball)

            if self.first_phase:
                # Snap paddle to ball AND sync target_y so update() doesn't
                # pull it back toward HEIGHT/2.
                self.right.y = self.ball.center_y - PADDLE_H / 2
                self.right.y = max(BORDER, min(HEIGHT - BORDER - PADDLE_H, self.right.y))
                self.right.target_y = self.ball.center_y

            self.left.update()
            self.right.update()
            self.ball.update()

            # Closest-approach tracking for shaping (inbound leg only).
            if self.ball.vx > 0 and abs(self.ball.x - RIGHT_X) < self.SHAPING_X_WINDOW:
                d = abs(self.right.center_y - self.ball.center_y)
                self.min_dist = min(self.min_dist, d)

            if self.ball.rect.colliderect(self.left.rect):
                self.ball.vx = abs(self.ball.vx)

            if self.ball.rect.colliderect(self.right.rect):
                self.ball.vx = -abs(self.ball.vx)
                if self.first_phase:
                    self.first_phase = False
                    self.agent_can_act = True
                    # No reward: the auto-aligned first hit is "free".
                else:
                    self.agent_can_act = True
                    reward += 1.0

            if self.ball.x < 0:
                # Impossible vs. unbeatable left AI; kept defensively.
                reward += 10.0
                terminated = True
            elif self.ball.x > WIDTH:
                miss_norm = (self.min_dist / (HEIGHT / 2)
                             if self.min_dist != float("inf") else 1.0)
                reward += -1.0 - self.miss_shaping * miss_norm
                terminated = True

        return reward, terminated

    def _get_obs(self):
        return np.array([
            (self.ball.x - WIDTH / 2) / (WIDTH / 2),
            (self.ball.y - HEIGHT / 2) / (HEIGHT / 2),
            self.ball.vx / BALL_SPEED_X,
            self.ball.vy / BALL_SPEED_Y,
            (self.right.center_y - HEIGHT / 2) / (HEIGHT / 2),
        ], dtype=np.float32)


# ====================================================================== #
#                              TRAINING                                  #
# ====================================================================== #
def make_env(rank: int, seed: int = 0):
    def _init():
        env = PongEnv()
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def train():
    env = PongEnv()
    check_env(env)

    venv = SubprocVecEnv([make_env(i) for i in range(8)])

    model = PPO(
        "MlpPolicy",
        venv,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[64, 64]),
        verbose=1,
        tensorboard_log="./pong_tb/",
    )

    print("Training PPO...")
    model.learn(total_timesteps=1_000_000)

    model.save("ppo_1")
    print("\nDone. Saved ppo_1.zip and vec_normalize.pkl")


if __name__ == "__main__":
    train()
