import gymnasium as gym
import numpy as np
import pygame
import cv2
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from pong import Paddle, Ball, unbeatable_ai, clamp, draw_background, FG, BALL_SIZE,\
    WIDTH, HEIGHT, BORDER, PADDLE_W, PADDLE_H, LEFT_X, RIGHT_X, BALL_SPEED_X, BALL_SPEED_Y
C_MIN = BORDER + PADDLE_H / 2
C_MAX = HEIGHT - BORDER - PADDLE_H / 2
class PongEnv(gym.Env):
    def __init__(self, max_episode_steps=1000):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(84, 84, 1),
            dtype=np.uint8
        )
        self.action_space = spaces.Discrete(3)
        pygame.init()
        self.screen = pygame.Surface((WIDTH, HEIGHT))

        self.left = Paddle(LEFT_X, HEIGHT / 2)
        self.right = Paddle(RIGHT_X, HEIGHT / 2)
        self.max_episode_steps = max_episode_steps
        self.ball = Ball()
        self.steps = 0
    
    def _get_obs(self):
        self._render_frame()
        surface_array = pygame.surfarray.array3d(self.screen)
        frame = np.transpose(surface_array, (1, 0, 2))
        
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        obs = np.expand_dims(resized, axis=-1)
        return obs.astype(np.uint8)

    def _render_frame(self):
        draw_background(self.screen)
        pygame.draw.rect(self.screen, FG, self.left.rect)
        pygame.draw.rect(self.screen, FG, self.right.rect)
        pygame.draw.rect(self.screen, FG, self.ball.rect)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.left = Paddle(LEFT_X, HEIGHT / 2)
        self.right = Paddle(RIGHT_X, HEIGHT / 2)
        self.ball = Ball()
        self.steps = 0

        return self._get_obs(), {}

    def step(self, action):
        if action == 1:  # UP
            self.right.target_y = self.right.center_y - HEIGHT
        elif action == 2:  # DOWN
            self.right.target_y = self.right.center_y + HEIGHT
        else:  # 0 or STAY
            self.right.target_y = self.right.center_y
        
        reward = 0.0
        terminated = False

        unbeatable_ai(self.left, self.ball)
        self.left.update()
        self.right.update()
        self.ball.update()

        if self.ball.rect.colliderect(self.left.rect):
            self.ball.vx = abs(self.ball.vx)

        if self.ball.rect.colliderect(self.right.rect):
            self.ball.vx = -abs(self.ball.vx)
            reward += 1.0

        if self.ball.x < 0:
            reward += 10.0
            terminated = True
        elif self.ball.x > WIDTH:
            reward += -1.0
            terminated = True

        self.steps += 1
        truncated = (not terminated) and self.steps >= self.max_episode_steps
        
        return self._get_obs(), reward, terminated, truncated, {}

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
    venv = VecMonitor(venv)
    venv = VecFrameStack(venv, n_stack=4)

    model = PPO(
        "CnnPolicy",
        venv,
        n_steps=512,
        batch_size=512,
        n_epochs=10,
        learning_rate=1e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./pong_tb_cnn/",
    )

    print("Training PPO...")
    model.learn(total_timesteps=1_000_000)

    model.save("ppo_cnn")
    print("\nDone. Saved ppo_cnn.zip")


if __name__ == "__main__":
    train()
