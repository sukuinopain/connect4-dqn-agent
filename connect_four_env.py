import numpy as np

class ConnectFour:
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        self.is_done = False
        self.winner = None

    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        self.is_done = False
        self.winner = None
        return self.get_state()

    def get_state(self):
        return self.board.copy()

    def get_available_actions(self):
        return [c for c in range(self.cols) if self.board[0, c] == 0]

    def check_winner(self):
        for r in range(self.rows):
            for c in range(self.cols - 3):
                window = self.board[r, c:c+4]
                if abs(sum(window)) == 4:
                    return int(window[0])

        for c in range(self.cols):
            for r in range(self.rows - 3):
                window = self.board[r:r+4, c]
                if abs(sum(window)) == 4:
                    return int(window[0])

        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                window = [self.board[r+i, c+i] for i in range(4)]
                if abs(sum(window)) == 4:
                    return int(window[0])

        for r in range(3, self.rows):
            for c in range(self.cols - 3):
                window = [self.board[r-i, c+i] for i in range(4)]
                if abs(sum(window)) == 4:
                    return int(window[0])

        if len(self.get_available_actions()) == 0:
            return 0

        return None

    def step(self, action):
        if self.is_done:
            raise ValueError("Game is already over.")

        if action not in self.get_available_actions():
            raise ValueError(f"Invalid move: Column {action} is full or out of range.")

        for r in range(self.rows - 1, -1, -1):
            if self.board[r, action] == 0:
                self.board[r, action] = self.current_player
                break

        result = self.check_winner()

        if result is not None:
            self.is_done = True
            self.winner = result
            if self.winner == self.current_player:
                reward = 1.0
            elif self.winner == 0:
                reward = 0.0
            else:
                reward = -1.0
        else:
            reward = 0.0

        self.current_player = -self.current_player
        return self.get_state(), reward, self.is_done, self.winner

    def render(self):
        symbols = {1: 'X', -1: 'O', 0: '.'}
        print("\n 0 1 2 3 4 5 6")
        for r in range(self.rows):
            print(' '.join(symbols[val] for val in self.board[r]))
        print("-" * 15)