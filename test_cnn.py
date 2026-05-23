import pygame
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from cnn_agent import Paddle, Ball, unbeatable_ai, clamp, draw_background, process_screen, cnn_ppo_model, random_agent_action, \
    WIDTH, HEIGHT, LEFT_X, RIGHT_X, BORDER, PADDLE_H, FG, C_MIN, C_MAX

def run_headless_test_cnn(num_episodes=100, max_hits=100):
    pygame.init()
    
    # 无头模式：在后台创建一个 Surface 用于绘制和截图，不需要 set_mode() 显示窗口
    screen = pygame.Surface((WIDTH, HEIGHT))

    left = Paddle(LEFT_X, HEIGHT / 2)
    right = Paddle(RIGHT_X, HEIGHT / 2)
    ball = Ball()

    hit_counts = []
    frame_stack = deque(maxlen=4)

    print(f"🚀 CNN Headless Test Start\n")

    for episode in range(num_episodes):
        left.y = left.target_y = HEIGHT / 2
        right.y = right.target_y = HEIGHT / 2
        ball.reset()

        agent_can_act = False
        first_phase = True
        current_hits = 0
        frame_stack.clear()

        while True:
            unbeatable_ai(left, ball)

            if first_phase:
                right.y = ball.center_y - PADDLE_H / 2
                right.y = clamp(right.y, BORDER, HEIGHT - BORDER - PADDLE_H)
                right.target_y = ball.center_y

            left.update()
            right.update()
            ball.update()

            if ball.rect.colliderect(left.rect):
                ball.vx = abs(ball.vx)

            if ball.rect.colliderect(right.rect):
                ball.vx = -abs(ball.vx)
                if first_phase:
                    first_phase = False
                else:
                    current_hits += 1

                agent_can_act = True

            # 提前截断逻辑
            if current_hits >= max_hits:
                hit_counts.append(current_hits)
                print(f"episode {episode + 1:03d} success | 达到目标击球数: {current_hits}")
                break

            if ball.x < 0:
                hit_counts.append(current_hits)
                print(f"episode {episode + 1:03d} agent beats unbeatable AI | 连续击球数: {current_hits}")
                break
            
            elif ball.x > WIDTH:
                hit_counts.append(current_hits)
                print(f"episode {episode + 1:03d} ended | 连续击球数: {current_hits}")
                break
                
            # -------- 渲染画面供 CNN 使用 --------
            draw_background(screen)
            pygame.draw.rect(screen, FG, left.rect)
            pygame.draw.rect(screen, FG, right.rect)
            pygame.draw.rect(screen, FG, ball.rect)
            
            current_frame = process_screen(screen)
            frame_stack.append(current_frame)
            
            # -------- Agent 动作逻辑 --------
            if agent_can_act:
                if len(frame_stack) == 4:
                    obs = np.stack(frame_stack, axis=-1)
                    if cnn_ppo_model is not None:
                        action, _ = cnn_ppo_model.predict(obs, deterministic=True)
                        target_y = float((action[0] + 1.0) / 2.0 * (C_MAX - C_MIN) + C_MIN)
                    else:
                        target_y = random_agent_action()
                else:
                    # 前几帧凑不齐的情况（由于 first phase 的存在，通常这里不会发生）
                    target_y = right.center_y
                
                right.target_y = target_y
                agent_can_act = False

    print("\n" + "="*40)
    print(f"num of episodes:   {num_episodes}")
    print(f"mean hit counts: {np.mean(hit_counts):.2f} 次")
    print(f"highest hit counts: {np.max(hit_counts)} 次")
    print(f"lowest hit counts: {np.min(hit_counts)} 次")
    print(f"std hit counts: {np.std(hit_counts):.2f}")
    print("="*40)

    # 绘制结果图表
    plt.figure(figsize=(12, 6))
    plt.bar(range(1, num_episodes + 1), hit_counts, color='mediumseagreen', alpha=0.7)
    plt.axhline(y=np.mean(hit_counts), color='red', linestyle='--', label=f'Mean: {np.mean(hit_counts):.2f}')
    plt.title('CNN Agent Test Results: Hits per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Hit Counts')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('cnn_test_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("📈 Plot saved as cnn_test_results.png\n")

if __name__ == "__main__":
    run_headless_test_cnn(100, max_hits=100)
