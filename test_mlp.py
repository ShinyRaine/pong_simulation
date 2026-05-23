import pygame
import numpy as np
import matplotlib.pyplot as plt

from mlp_agent import Paddle, Ball, unbeatable_ai, get_agent_action, clamp, \
 WIDTH, HEIGHT, LEFT_X, RIGHT_X, BORDER, PADDLE_H

def run_headless_test(num_episodes=100, max_hits=100):
    pygame.init()

    left = Paddle(LEFT_X, HEIGHT / 2)
    right = Paddle(RIGHT_X, HEIGHT / 2)
    ball = Ball()

    hit_counts = []

    print(f"🚀 test start\n")

    for episode in range(num_episodes):
        left.y = left.target_y = HEIGHT / 2
        right.y = right.target_y = HEIGHT / 2
        ball.reset()

        agent_can_act = False
        first_phase = True
        current_hits = 0

        while True:
            unbeatable_ai(left, ball)

            if first_phase:
                right.y = ball.center_y - PADDLE_H / 2
                right.y = clamp(right.y, BORDER, HEIGHT - BORDER - PADDLE_H)
            else:
                if agent_can_act:
                    right.target_y = get_agent_action(ball, right)
                    agent_can_act = False

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

    print("\n" + "="*40)
    print(f"num of episodes:   {num_episodes}")
    print(f"mean hit counts: {np.mean(hit_counts):.2f} 次")
    print(f"heighest hit counts: {np.max(hit_counts)} 次")
    print(f"lowest hit counts: {np.min(hit_counts)} 次")
    print(f"std hit counts: {np.std(hit_counts):.2f}")
    print("="*40)

    # 绘制结果图表
    plt.figure(figsize=(12, 6))
    plt.bar(range(1, num_episodes + 1), hit_counts, color='royalblue', alpha=0.7)
    plt.axhline(y=np.mean(hit_counts), color='red', linestyle='--', label=f'Mean: {np.mean(hit_counts):.2f}')
    plt.title('MLP Agent Test Results: Hits per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Hit Counts')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('mlp_test_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("📈 Plot saved as mlp_test_results.png\n")

if __name__ == "__main__":
    run_headless_test(100)
