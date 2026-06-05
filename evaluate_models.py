import pygame
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from match import (
    MLPAgent, CNNAgent, Paddle, Ball, clamp,
    WIDTH, HEIGHT, BORDER, PADDLE_H, FG, LEFT_X, RIGHT_X, draw_background
)

def run_match_eval():
    pygame.init()
    # Use hidden display for fast headless execution
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.HIDDEN)
    score_font = pygame.font.SysFont("consolas", 60)
    label_font = pygame.font.SysFont("consolas", 22)

    mlp_path = "output/ppo_mlp_1M.zip"
    cnn_path = "output/ppo_cnn_1M.zip"

    print("Loading models...")
    left_agent = MLPAgent(mlp_path, side="left")
    right_agent = CNNAgent(cnn_path, side="right")
    left_name, right_name = "MLP", "CNN"

    num_matches = 1
    # 5 minutes of game time at 60 FPS = 5 * 60 * 60 = 18000 frames
    max_frames = 18000

    mlp_wins = 0
    cnn_wins = 0

    results = []

    print(f"Starting {num_matches} matches, each lasting {max_frames} frames (5 mins game time)...")
    start_time = time.time()

    for match_idx in range(num_matches):
        print(f"\n--- Starting Match {match_idx + 1}/{num_matches} ---")
        left = Paddle(LEFT_X, HEIGHT / 2)
        right = Paddle(RIGHT_X, HEIGHT / 2)
        left.target_y = HEIGHT / 2
        right.target_y = HEIGHT / 2
        ball = Ball()

        left_score = 0
        right_score = 0

        first_phase_right = True
        first_phase_left = True
        right_can_act = False
        left_can_act = False

        if isinstance(left_agent, CNNAgent):
            left_agent.reset_state()
        if isinstance(right_agent, CNNAgent):
            right_agent.reset_state()

        def reset_after_score():
            nonlocal first_phase_right, first_phase_left, right_can_act, left_can_act
            ball.reset()
            first_phase_right = True
            first_phase_left = True
            right_can_act = False
            left_can_act = False
            if isinstance(left_agent, CNNAgent):
                left_agent.reset_state()
            if isinstance(right_agent, CNNAgent):
                right_agent.reset_state()

        frames = 0
        while frames < max_frames:
            # decision phase
            if first_phase_right:
                right.y = ball.center_y - PADDLE_H / 2
                right.y = clamp(right.y, BORDER, HEIGHT - BORDER - PADDLE_H)
                right.target_y = ball.center_y
            elif right_can_act:
                right.target_y = right_agent.act(left, right, ball) if isinstance(right_agent, CNNAgent) else right_agent.act(ball, left, right)
                right_can_act = False

            if first_phase_left:
                left.y = ball.center_y - PADDLE_H / 2
                left.y = clamp(left.y, BORDER, HEIGHT - BORDER - PADDLE_H)
                left.target_y = ball.center_y
            elif left_can_act:
                left.target_y = left_agent.act(left, right, ball) if isinstance(left_agent, CNNAgent) else left_agent.act(ball, left, right)
                left_can_act = False

            # physics
            left.update()
            right.update()
            ball.update()

            # collisions
            if ball.rect.colliderect(left.rect):
                ball.vx = abs(ball.vx)
                if first_phase_left: first_phase_left = False
                left_can_act = True

            if ball.rect.colliderect(right.rect):
                ball.vx = -abs(ball.vx)
                if first_phase_right: first_phase_right = False
                right_can_act = True

            if isinstance(left_agent, CNNAgent): left_agent.record_frame(left, right, ball)
            if isinstance(right_agent, CNNAgent): right_agent.record_frame(left, right, ball)

            # score
            if ball.x < 0:
                right_score += 1
                reset_after_score()
            elif ball.x > WIDTH:
                left_score += 1
                reset_after_score()

            frames += 1

            if frames % 3000 == 0:
                print(f"  Frame {frames}/{max_frames} - Score: MLP {left_score} - {right_score} CNN")

            # Save screenshot at the very end of the FIRST match
            if match_idx == 0 and frames == max_frames:
                draw_background(screen)
                pygame.draw.rect(screen, FG, left.rect)
                pygame.draw.rect(screen, FG, right.rect)
                pygame.draw.rect(screen, FG, ball.rect)

                ls = score_font.render(str(left_score), True, FG)
                rs = score_font.render(str(right_score), True, FG)
                screen.blit(ls, (WIDTH // 2 - 120, 20))
                screen.blit(rs, (WIDTH // 2 + 60, 20))

                ll = label_font.render(left_name, True, (160, 160, 160))
                rl = label_font.render(right_name, True, (160, 160, 160))
                screen.blit(ll, (20, HEIGHT - 30))
                screen.blit(rl, (WIDTH - rl.get_width() - 20, HEIGHT - 30))
                
                pygame.display.flip()
                pygame.image.save(screen, "eval_screenshot.png")
                print(">> Saved screenshot of match 1 to eval_screenshot.png")

        print(f"Match {match_idx + 1} finished: MLP {left_score} - {right_score} CNN")
        results.append((left_score, right_score))
        if left_score > right_score:
            mlp_wins += 1
        elif right_score > left_score:
            cnn_wins += 1

    total_time = time.time() - start_time
    print(f"\nEvaluation Complete! (Took {total_time:.1f} seconds)")
    print(f"MLP Wins: {mlp_wins}, CNN Wins: {cnn_wins}, Draws: {num_matches - mlp_wins - cnn_wins}")

    # Plot results
    labels = ['MLP Wins', 'CNN Wins', 'Draws']
    sizes = [mlp_wins, cnn_wins, num_matches - mlp_wins - cnn_wins]
    colors = ['#ff9999','#66b3ff','#99ff99']
    
    # Filter out 0 sizes
    sizes_filtered = []
    labels_filtered = []
    colors_filtered = []
    for s, l, c in zip(sizes, labels, colors):
        if s > 0:
            sizes_filtered.append(s)
            labels_filtered.append(l)
            colors_filtered.append(c)

    plt.figure(figsize=(8, 6))
    plt.pie(sizes_filtered, labels=labels_filtered, colors=colors_filtered, autopct='%1.1f%%', startangle=140)
    plt.title('Win Rate over 1 Match (5 mins)')
    plt.axis('equal')
    plt.savefig('win_rate_chart.png')
    print(">> Saved win rate chart to win_rate_chart.png")

if __name__ == "__main__":
    run_match_eval()
