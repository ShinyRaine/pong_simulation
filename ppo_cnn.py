import gymnasium as gym
import numpy as np
import pygame
import cv2
import collections
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
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
            shape=(84, 84, 4),
            dtype=np.uint8
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        pygame.init()
        self.screen = pygame.Surface((WIDTH, HEIGHT))

        self.left = Paddle(LEFT_X, HEIGHT / 2)
        self.right = Paddle(RIGHT_X, HEIGHT / 2)
        self.max_episode_steps = max_episode_steps
        self.ball = Ball()
        self.first_phase = True
        self.agent_can_act = False
        self.steps = 0
        self.frames = collections.deque(maxlen=4)
    
    def _get_gray_frame(self):
        self._render_frame()
        surface_array = pygame.surfarray.array3d(self.screen)
        frame = np.transpose(surface_array, (1, 0, 2))
        
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def _get_obs(self):
        obs = np.stack(list(self.frames), axis=-1)
        return obs.astype(np.uint8)

    def _render_frame(self):
        draw_background(self.screen)
        pygame.draw.rect(self.screen, FG, self.left.rect)
        pygame.draw.rect(self.screen, FG, self.right.rect)
        pygame.draw.rect(self.screen, FG, self.ball.rect)

    def _simulate_until_action(self):
        reward = 0.0
        terminated = False
        min_dist = float("inf")

        while not self.agent_can_act and not terminated:
            unbeatable_ai(self.left, self.ball)

            if self.first_phase:
                self.right.y = self.ball.center_y - PADDLE_H / 2
                self.right.y = clamp(self.right.y, BORDER, HEIGHT - BORDER - PADDLE_H)
                self.right.target_y = self.ball.center_y

            self.left.update()
            self.right.update()
            self.ball.update()

            if self.ball.rect.colliderect(self.left.rect):
                self.ball.vx = abs(self.ball.vx)

            if self.ball.rect.colliderect(self.right.rect):
                self.ball.vx = -abs(self.ball.vx)
                if self.first_phase:
                    self.first_phase = False
                    self.agent_can_act = True
                else:
                    self.agent_can_act = True
                    reward += 1.0

            if self.ball.vx > 0:
                d = abs(self.right.center_y - self.ball.center_y)
                min_dist = min(min_dist, d)

            if self.ball.x < 0:
                reward += 10.0
                terminated = True
            elif self.ball.x > WIDTH:
                reward -= 1.0
                terminated = True
                
            self.frames.append(self._get_gray_frame())

        return reward, terminated

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.left = Paddle(LEFT_X, HEIGHT / 2)
        self.right = Paddle(RIGHT_X, HEIGHT / 2)
        self.ball = Ball()
        self.first_phase = True
        self.agent_can_act = False
        self.steps = 0
        
        initial_frame = self._get_gray_frame()
        for _ in range(4):
            self.frames.append(initial_frame)
            
        _, terminated = self._simulate_until_action()

        return self._get_obs(), {}

    def step(self, action):
        a = float(np.clip(action[0], -1.0, 1.0))
        target_y = (a + 1.0) / 2.0 * (C_MAX - C_MIN) + C_MIN
        self.right.target_y = target_y
        self.agent_can_act = False
        
        reward, terminated = self._simulate_until_action()

        self.steps += 1
        truncated = (not terminated) and self.steps >= self.max_episode_steps
        
        return self._get_obs(), reward, terminated, truncated, {}

from typing import Callable
def exp_schedule(initial_value: float, decay_rate: float = 0.95) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return initial_value * (decay_rate ** ((1.0 - progress_remaining) * 10)) 
    return func

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

    eval_venv = SubprocVecEnv([make_env(i + 100) for i in range(8)])
    eval_venv = VecMonitor(eval_venv)

    stop_train_callback = StopTrainingOnRewardThreshold(reward_threshold=9.0, verbose=1)

    eval_callback = EvalCallback(
        eval_venv, 
        callback_on_new_best=stop_train_callback, 
        eval_freq=500, 
        n_eval_episodes=100, 
        best_model_save_path='./logs/best_cnn/', 
        verbose=1
    )

    model = PPO(
        "CnnPolicy",
        venv,
        n_steps=512,
        batch_size=512,
        n_epochs=10,
        learning_rate=exp_schedule(1e-4),
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./pong_tb_cnn/500fq",
    )

    print("Training PPO...")
    model.learn(total_timesteps=5_000_000, callback=eval_callback)

    model.save("ppo_cnn_500fq")
    print("\nDone. Saved ppo_cnn_500fq.zip")


if __name__ == "__main__":
    train()
