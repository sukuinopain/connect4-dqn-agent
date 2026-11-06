import numpy as np
from connect_four_env import ConnectFour
from dqn_agent import DQNAgent

class PureMinimaxAgent:
    def __init__(self, depth=4):
        self.depth = depth

    def get_drop_row(self, board, col):
        for r in range(5, -1, -1):
            if board[r, col] == 0:
                return r
        return -1

    def check_win(self, board, r, c, player):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for step in range(1, 4):
                nr, nc = r + dr * step, c + dc * step
                if 0 <= nr < 6 and 0 <= nc < 7 and board[nr, nc] == player:
                    count += 1
                else:
                    break
            for step in range(1, 4):
                nr, nc = r - dr * step, c - dc * step
                if 0 <= nr < 6 and 0 <= nc < 7 and board[nr, nc] == player:
                    count += 1
                else:
                    break
            if count >= 4:
                return True
        return False

    def score_window(self, window, player):
        score = 0
        opp = -player
        p_cnt = np.count_nonzero(window == player)
        o_cnt = np.count_nonzero(window == opp)
        e_cnt = np.count_nonzero(window == 0)

        if p_cnt == 4:
            score += 1000
        elif p_cnt == 3 and e_cnt == 1:
            score += 10
        elif p_cnt == 2 and e_cnt == 2:
            score += 4

        if o_cnt == 3 and e_cnt == 1:
            score -= 80
        elif o_cnt == 2 and e_cnt == 2:
            score -= 10

        return score

    def evaluate_board(self, board, player):
        score = 0
        center_col = board[:, 3]
        score += np.count_nonzero(center_col == player) * 6
        score -= np.count_nonzero(center_col == -player) * 6

        for r in range(6):
            for c in range(4):
                score += self.score_window(board[r, c:c+4], player)

        for c in range(7):
            for r in range(3):
                score += self.score_window(board[r:r+4, c], player)

        for r in range(3):
            for c in range(4):
                w1 = np.array([board[r+i, c+i] for i in range(4)])
                w2 = np.array([board[r+3-i, c+i] for i in range(4)])
                score += self.score_window(w1, player)
                score += self.score_window(w2, player)

        return score

    def minimax(self, board, depth, alpha, beta, is_maximizing, player):
        available = [c for c in range(7) if board[0, c] == 0]
        ordered = sorted(available, key=lambda c: abs(c - 3))

        if not ordered:
            return 0, None

        if depth == 0:
            return self.evaluate_board(board, player), None

        if is_maximizing:
            for a in ordered:
                r = self.get_drop_row(board, a)
                board[r, a] = player
                if self.check_win(board, r, a, player):
                    board[r, a] = 0
                    return 100000 + depth, a
                board[r, a] = 0

            max_eval = -float('inf')
            best_action = ordered[0]
            for a in ordered:
                r = self.get_drop_row(board, a)
                board[r, a] = player
                score, _ = self.minimax(board, depth - 1, alpha, beta, False, player)
                board[r, a] = 0
                if score > max_eval:
                    max_eval = score
                    best_action = a
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
            return max_eval, best_action
        else:
            opp = -player
            for a in ordered:
                r = self.get_drop_row(board, a)
                board[r, a] = opp
                if self.check_win(board, r, a, opp):
                    board[r, a] = 0
                    return -100000 - depth, a
                board[r, a] = 0

            min_eval = float('inf')
            best_action = ordered[0]
            for a in ordered:
                r = self.get_drop_row(board, a)
                board[r, a] = opp
                score, _ = self.minimax(board, depth - 1, alpha, beta, True, player)
                board[r, a] = 0
                if score < min_eval:
                    min_eval = score
                    best_action = a
                beta = min(beta, score)
                if beta <= alpha:
                    break
            return min_eval, best_action

    def choose_action(self, board, player):
        board_copy = board.copy()
        _, action = self.minimax(board_copy, self.depth, -float('inf'), float('inf'), True, player)
        return action

def evaluate_dqn_vs_minimax(games=20):
    env = ConnectFour()
    dqn_agent = DQNAgent(epsilon=0.0)
    dqn_agent.load_model("dqn_connect4.pth")
    minimax_agent = PureMinimaxAgent(depth=4)

    print(f"\n--- Round 1: DQN (Player 1 - X) vs Pure Minimax (Player 2 - O) [{games} games] ---")
    dqn_wins = 0
    minimax_wins = 0
    draws = 0

    for g in range(1, games + 1):
        state = env.reset()
        done = False
        while not done:
            avail = env.get_available_actions()
            if env.current_player == 1:
                action = dqn_agent.choose_action(state, avail, search_depth=4)
            else:
                action = minimax_agent.choose_action(state, -1)
            state, _, done, winner = env.step(action)

        if winner == 1:
            dqn_wins += 1
            res = "DQN Won"
        elif winner == -1:
            minimax_wins += 1
            res = "Minimax Won"
        else:
            draws += 1
            res = "Draw"
        print(f"Game {g}/{games}: {res}")

    print(f"Results -> DQN: {dqn_wins} | Minimax: {minimax_wins} | Draws: {draws}")

    print(f"\n--- Round 2: Pure Minimax (Player 1 - X) vs DQN (Player 2 - O) [{games} games] ---")
    dqn_wins_2 = 0
    minimax_wins_2 = 0
    draws_2 = 0

    for g in range(1, games + 1):
        state = env.reset()
        done = False
        while not done:
            avail = env.get_available_actions()
            if env.current_player == 1:
                action = minimax_agent.choose_action(state, 1)
            else:
                norm_state = state * -1
                action = dqn_agent.choose_action(norm_state, avail, search_depth=4)
            state, _, done, winner = env.step(action)

        if winner == -1:
            dqn_wins_2 += 1
            res = "DQN Won"
        elif winner == 1:
            minimax_wins_2 += 1
            res = "Minimax Won"
        else:
            draws_2 += 1
            res = "Draw"
        print(f"Game {g}/{games}: {res}")

    print(f"Results -> DQN: {dqn_wins_2} | Minimax: {minimax_wins_2} | Draws: {draws_2}")

if __name__ == "__main__":
    evaluate_dqn_vs_minimax(games=20)