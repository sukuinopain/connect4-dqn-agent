import sys
import math
import pygame
import numpy as np
from connect_four_env import ConnectFour
from dqn_agent import DQNAgent

SQUARESIZE = 100
RADIUS = int(SQUARESIZE / 2 - 6)
WIDTH = 7 * SQUARESIZE
HEIGHT = (6 + 1) * SQUARESIZE
SIZE = (WIDTH, HEIGHT)

COLOR_BOARD = (30, 60, 114)
COLOR_BG = (15, 23, 42)
COLOR_EMPTY = (30, 41, 59)
COLOR_RED = (239, 68, 68)
COLOR_YELLOW = (234, 179, 8)
COLOR_TEXT = (248, 250, 252)
COLOR_BUTTON = (59, 130, 246)
COLOR_BUTTON_HOVER = (37, 99, 235)

STATE_MENU = 0
STATE_PLAYING = 1
STATE_GAMEOVER = 2

def draw_button(screen, rect, text, font, is_hovered):
    color = COLOR_BUTTON_HOVER if is_hovered else COLOR_BUTTON
    pygame.draw.rect(screen, color, rect, border_radius=12)
    pygame.draw.rect(screen, COLOR_TEXT, rect, width=2, border_radius=12)
    text_surf = font.render(text, True, COLOR_TEXT)
    text_rect = text_surf.get_rect(center=rect.center)
    screen.blit(text_surf, text_rect)

def draw_menu(screen, title_font, btn_font, mouse_pos):
    screen.fill(COLOR_BG)

    title_surf = title_font.render("CONNECT FOUR AI", True, COLOR_TEXT)
    title_rect = title_surf.get_rect(center=(WIDTH // 2, HEIGHT // 3 - 30))
    screen.blit(title_surf, title_rect)

    subtitle_font = pygame.font.SysFont("Segoe UI", 24)
    sub_surf = subtitle_font.render("Choose your turn to start:", True, (148, 163, 184))
    sub_rect = sub_surf.get_rect(center=(WIDTH // 2, HEIGHT // 3 + 25))
    screen.blit(sub_surf, sub_rect)

    btn1_rect = pygame.Rect(WIDTH // 2 - 160, HEIGHT // 2, 320, 60)
    btn2_rect = pygame.Rect(WIDTH // 2 - 160, HEIGHT // 2 + 80, 320, 60)

    draw_button(screen, btn1_rect, "Play First (Red)", btn_font, btn1_rect.collidepoint(mouse_pos))
    draw_button(screen, btn2_rect, "Play Second (Yellow)", btn_font, btn2_rect.collidepoint(mouse_pos))

    pygame.display.update()
    return btn1_rect, btn2_rect

def draw_game(screen, board, font, status_text, hover_col=None, current_turn_color=None):
    screen.fill(COLOR_BG)

    if hover_col is not None and current_turn_color is not None:
        center_x = int(hover_col * SQUARESIZE + SQUARESIZE / 2)
        center_y = int(SQUARESIZE / 2)
        pygame.draw.circle(screen, current_turn_color, (center_x, center_y), RADIUS)

    if status_text:
        text_surf = font.render(status_text, True, COLOR_TEXT)
        text_rect = text_surf.get_rect(center=(WIDTH // 2, SQUARESIZE // 2))
        screen.blit(text_surf, text_rect)

    for c in range(7):
        for r in range(6):
            rect = (c * SQUARESIZE, (r + 1) * SQUARESIZE, SQUARESIZE, SQUARESIZE)
            pygame.draw.rect(screen, COLOR_BOARD, rect)

            center = (int(c * SQUARESIZE + SQUARESIZE / 2), int((r + 1) * SQUARESIZE + SQUARESIZE / 2))
            val = board[r, c]
            if val == 1:
                color = COLOR_RED
            elif val == -1:
                color = COLOR_YELLOW
            else:
                color = COLOR_EMPTY
            pygame.draw.circle(screen, color, center, RADIUS)

    pygame.display.update()

def draw_game_over(screen, board, font, btn_font, result_text, mouse_pos):
    screen.fill(COLOR_BG)

    for c in range(7):
        for r in range(6):
            rect = (c * SQUARESIZE, (r + 1) * SQUARESIZE, SQUARESIZE, SQUARESIZE)
            pygame.draw.rect(screen, COLOR_BOARD, rect)

            center = (int(c * SQUARESIZE + SQUARESIZE / 2), int((r + 1) * SQUARESIZE + SQUARESIZE / 2))
            val = board[r, c]
            if val == 1:
                color = COLOR_RED
            elif val == -1:
                color = COLOR_YELLOW
            else:
                color = COLOR_EMPTY
            pygame.draw.circle(screen, color, center, RADIUS)

    overlay = pygame.Surface((WIDTH, SQUARESIZE))
    overlay.fill(COLOR_BG)
    screen.blit(overlay, (0, 0))

    text_surf = font.render(result_text, True, COLOR_TEXT)
    text_rect = text_surf.get_rect(center=(WIDTH // 2, SQUARESIZE // 2))
    screen.blit(text_surf, text_rect)

    btn_restart = pygame.Rect(WIDTH // 2 - 210, HEIGHT - 70, 200, 50)
    btn_menu = pygame.Rect(WIDTH // 2 + 10, HEIGHT - 70, 200, 50)

    draw_button(screen, btn_restart, "Play Again", btn_font, btn_restart.collidepoint(mouse_pos))
    draw_button(screen, btn_menu, "Main Menu", btn_font, btn_menu.collidepoint(mouse_pos))

    pygame.display.update()
    return btn_restart, btn_menu

def run_gui():
    pygame.init()
    screen = pygame.display.set_mode(SIZE)
    pygame.display.set_caption("Connect Four: Master AI")
    
    title_font = pygame.font.SysFont("Segoe UI", 44, bold=True)
    status_font = pygame.font.SysFont("Segoe UI", 30, bold=True)
    btn_font = pygame.font.SysFont("Segoe UI", 22, bold=True)

    env = ConnectFour()
    agent = DQNAgent(epsilon=0.0)
    agent.load_model("dqn_connect4.pth")

    game_state = STATE_MENU
    human_player = 1
    agent_player = -1
    hover_col = 3
    result_text = ""

    btn_menu_1 = None
    btn_menu_2 = None
    btn_restart = None
    btn_menu_back = None

    while True:
        mouse_pos = pygame.mouse.get_pos()

        if game_state == STATE_MENU:
            btn_menu_1, btn_menu_2 = draw_menu(screen, title_font, btn_font, mouse_pos)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if btn_menu_1.collidepoint(mouse_pos):
                        human_player = 1
                        agent_player = -1
                        env.reset()
                        game_state = STATE_PLAYING
                    elif btn_menu_2.collidepoint(mouse_pos):
                        human_player = -1
                        agent_player = 1
                        env.reset()
                        game_state = STATE_PLAYING
                        draw_game(screen, env.board, status_font, "AI Thinking...")
                        pygame.time.wait(200)
                        norm_state = env.board * agent_player
                        action = agent.choose_action(norm_state, env.get_available_actions(), search_depth=4)
                        env.step(action)

        elif game_state == STATE_PLAYING:
            human_turn = (env.current_player == human_player)
            current_color = COLOR_RED if human_player == 1 else COLOR_YELLOW
            status_text = "Your Turn" if human_turn else "AI Thinking..."

            draw_game(screen, env.board, status_font, status_text, hover_col if human_turn else None, current_color)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.MOUSEMOTION:
                    col = int(math.floor(event.pos[0] / SQUARESIZE))
                    hover_col = max(0, min(6, col))

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and human_turn:
                    col = int(math.floor(event.pos[0] / SQUARESIZE))
                    avail = env.get_available_actions()

                    if col in avail:
                        _, _, done, winner = env.step(col)

                        if done:
                            game_state = STATE_GAMEOVER
                            if winner == human_player:
                                result_text = "Congratulations! You Won!"
                            elif winner == 0:
                                result_text = "Game Over! It's a Draw!"
                            else:
                                result_text = "AI Won! Better luck next time."
                        else:
                            draw_game(screen, env.board, status_font, "AI Thinking...")
                            pygame.time.wait(150)
                            agent_avail = env.get_available_actions()
                            norm_state = env.board * agent_player
                            ai_action = agent.choose_action(norm_state, agent_avail, search_depth=4)
                            _, _, ai_done, ai_winner = env.step(ai_action)

                            if ai_done:
                                game_state = STATE_GAMEOVER
                                if ai_winner == agent_player:
                                    result_text = "AI Won! Better luck next time."
                                elif ai_winner == 0:
                                    result_text = "Game Over! It's a Draw!"
                                else:
                                    result_text = "Congratulations! You Won!"

        elif game_state == STATE_GAMEOVER:
            btn_restart, btn_menu_back = draw_game_over(screen, env.board, status_font, btn_font, result_text, mouse_pos)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if btn_restart.collidepoint(mouse_pos):
                        env.reset()
                        game_state = STATE_PLAYING
                        if human_player == -1:
                            draw_game(screen, env.board, status_font, "AI Thinking...")
                            pygame.time.wait(200)
                            norm_state = env.board * agent_player
                            action = agent.choose_action(norm_state, env.get_available_actions(), search_depth=4)
                            env.step(action)
                    elif btn_menu_back.collidepoint(mouse_pos):
                        game_state = STATE_MENU

if __name__ == "__main__":
    run_gui()