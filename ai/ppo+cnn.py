"""
PPO + CNN training, two-snapshot variant.

Observation = (2, 84, 84) uint8:
    channel 0 = scene at the moment LEFT paddle hit the ball
    channel 1 = scene at the moment RIGHT paddle hit the ball (= now)

Compared to the "N physics frames ago" variant, this version gives
the CNN a much larger spatial separation between channels (the ball
moves across nearly the whole screen between hits), and the two
frames have clear semantic meaning: "ball was sent back from here"
vs. "ball arrived here."
"""

import random
import numpy as np

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from pong import (
    Paddle, Ball, unbeatable_ai,
    WIDTH, HEIGHT, BORDER, PADDLE_H, PADDLE_W,
    LEFT_X, RIGHT_X, BALL_SIZE,
)


class PongCNNEnv(gym.Env):
    metadata = {"render_modes": []}

    IMG_H = 84
    IMG_W = 84

    C_MIN = BORDER + PADDLE_H / 2
    C_MAX = HEIGHT - BORDER - PADDLE_H / 2

    def __init__(self, max_episode_steps: int = 200, miss_shaping: float = 0.5):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(2, self.IMG_H, self.IMG_W), dtype=np.uint8
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self.max_episode_steps = max_episode_steps
        self.miss_shaping = miss_shaping

        self.left = Paddle(LEFT_X, HEIGHT / 2)
        self.right = Paddle(RIGHT_X, HEIGHT / 2)
        self.ball = Ball()
        # Snapshot captured at the latest left-paddle hit. None until first hit.
        self.last_left_hit_snap = None
        self.first_phase = True
        self.agent_can_act = False
        self.steps = 0
        self.min_dist = float("inf")

    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)

        self.left = Paddle(LEFT_X, HEIGHT / 2)
        self.right = Paddle(RIGHT_X, HEIGHT / 2)
        self.left.target_y = HEIGHT / 2
        self.right.target_y = HEIGHT / 2
        self.ball = Ball()

        self.last_left_hit_snap = None
        self.first_phase = True
        self.agent_can_act = False
        self.steps = 0
        self.min_dist = float("inf")

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
        reward = 0.0
        terminated = False

        while not self.agent_can_act and not terminated:
            unbeatable_ai(self.left, self.ball)

            if self.first_phase:
                self.right.y = self.ball.center_y - PADDLE_H / 2
                self.right.y = max(BORDER, min(HEIGHT - BORDER - PADDLE_H, self.right.y))
                self.right.target_y = self.ball.center_y

            self.left.update()
            self.right.update()
            self.ball.update()

            if self.ball.vx > 0 and abs(self.ball.x - RIGHT_X) < 40:
                d = abs(self.right.center_y - self.ball.center_y)
                self.min_dist = min(self.min_dist, d)

            # LEFT collision — capture snapshot for channel 0.
            if self.ball.rect.colliderect(self.left.rect):
                self.ball.vx = abs(self.ball.vx)
                self.last_left_hit_snap = self._snapshot()

            # RIGHT collision — ends a decision cycle.
            if self.ball.rect.colliderect(self.right.rect):
                self.ball.vx = -abs(self.ball.vx)
                if self.first_phase:
                    self.first_phase = False
                    self.agent_can_act = True
                else:
                    self.agent_can_act = True
                    reward += 1.0

            if self.ball.x < 0:
                reward += 10.0
                terminated = True
            elif self.ball.x > WIDTH:
                miss_norm = (self.min_dist / (HEIGHT / 2)
                             if self.min_dist != float("inf") else 1.0)
                reward += -1.0 - self.miss_shaping * miss_norm
                terminated = True

        return reward, terminated

    # ------------------------------------------------------------------ #
    def _snapshot(self):
        return (self.ball.x, self.ball.y, self.left.y, self.right.y)

    def _render_frame(self, snap):
        ball_x, ball_y, left_y, right_y = snap
        frame = np.zeros((self.IMG_H, self.IMG_W), dtype=np.uint8)

        sx = self.IMG_W / WIDTH
        sy = self.IMG_H / HEIGHT

        bh = max(1, int(BORDER * sy))
        frame[:bh, :] = 80
        frame[-bh:, :] = 80

        pw = max(1, int(PADDLE_W * sx))
        ph = max(2, int(PADDLE_H * sy))

        lx = int(LEFT_X * sx)
        ly = int(np.clip(left_y * sy, 0, self.IMG_H - ph))
        frame[ly:ly + ph, lx:lx + pw] = 255

        rx = int(RIGHT_X * sx)
        ry = int(np.clip(right_y * sy, 0, self.IMG_H - ph))
        frame[ry:ry + ph, rx:rx + pw] = 255

        bs = max(2, int(BALL_SIZE * sx))
        bx = int(np.clip(ball_x * sx, 0, self.IMG_W - bs))
        by = int(np.clip(ball_y * sy, 0, self.IMG_H - bs))
        frame[by:by + bs, bx:bx + bs] = 255

        return frame

    def _get_obs(self):
        current = self._snapshot()
        # On the very first decision after reset, no left hit has occurred yet
        # (first_phase ends right after the FIRST right hit). Fall back to using
        # the current snapshot for both channels — happens once per episode.
        old = self.last_left_hit_snap if self.last_left_hit_snap is not None else current
        return np.stack(
            [self._render_frame(old), self._render_frame(current)], axis=0
        )


# ====================================================================== #
def make_env(rank: int, seed: int = 0):
    def _init():
        env = PongCNNEnv()
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def train():
    print("Checking env API...")
    check_env(PongCNNEnv())
    print("OK.\n")

    N_ENVS = 8
    venv = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])

    model = PPO(
        "CnnPolicy",
        venv,
        n_steps=512,
        batch_size=256,
        n_epochs=4,
        learning_rate=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
        verbose=1,
        policy_kwargs=dict(net_arch=[64, 64]),
        tensorboard_log="./pong_tb_cnn_hits/",
    )

    print("Training PPO + CNN (left-hit + right-hit obs)...")
    model.learn(total_timesteps=2_000_000)

    model.save("ppo_pong_cnn_hits")
    venv.close()
    print("\nDone. Saved ppo_pong_cnn_hits.zip")


def evaluate(n_episodes: int = 10):
    model = PPO.load("ppo_pong_cnn_hits")
    env = PongCNNEnv()

    ep_lens = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        ep_lens.append(steps)
        print(f"Episode {ep + 1:2d}: {steps:3d} successful returns")

    print(f"\nMean: {np.mean(ep_lens):.1f}  Max: {max(ep_lens)}  Min: {min(ep_lens)}")


if __name__ == "__main__":
    train()
    print("\n" + "=" * 60)
    print("Evaluating trained model...")
    print("=" * 60)
    evaluate(n_episodes=10)
