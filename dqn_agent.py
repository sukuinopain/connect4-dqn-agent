import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.bool_),
        )

    def __len__(self):
        return len(self.buffer)

class ConvQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

class DQNAgent:
    def __init__(
        self,
        learning_rate=0.0005,
        gamma=0.99,
        epsilon=0.0,
        epsilon_decay=0.999,
        min_epsilon=0.02,
        batch_size=64,
        buffer_capacity=50000,
        target_update_freq=250
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_steps = 0

        self.policy_net = ConvQNetwork().to(self.device)
        self.target_net = ConvQNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()
        self.memory = ReplayBuffer(buffer_capacity)

    def encode_state(self, state):
        ch1 = (state == 1).astype(np.float32)
        ch2 = (state == -1).astype(np.float32)
        return np.stack([ch1, ch2], axis=0)

    def encode_batch(self, states):
        ch1 = (states == 1).astype(np.float32)
        ch2 = (states == -1).astype(np.float32)
        return np.stack([ch1, ch2], axis=1)

    def get_drop_row(self, board, col):
        for r in range(5, -1, -1):
            if board[r, col] == 0:
                return r
        return -1

    def check_line_of_four(self, board, r, c, player):
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

    def score_window(self, window):
        score = 0.0
        agent_count = np.count_nonzero(window == 1)
        opp_count = np.count_nonzero(window == -1)
        empty_count = np.count_nonzero(window == 0)

        if agent_count == 3 and empty_count == 1:
            score += 5.0
        elif agent_count == 2 and empty_count == 2:
            score += 1.5

        if opp_count == 3 and empty_count == 1:
            score -= 6.0
        elif opp_count == 2 and empty_count == 2:
            score -= 1.8

        return score

    def evaluate_board(self, board):
        score = 0.0
        center_col = board[:, 3]
        score += np.count_nonzero(center_col == 1) * 2.0
        score -= np.count_nonzero(center_col == -1) * 2.0

        for r in range(6):
            for c in range(4):
                score += self.score_window(board[r, c:c+4])

        for c in range(7):
            for r in range(3):
                score += self.score_window(board[r:r+4, c])

        for r in range(3):
            for c in range(4):
                w1 = np.array([board[r+i, c+i] for i in range(4)])
                w2 = np.array([board[r+3-i, c+i] for i in range(4)])
                score += self.score_window(w1)
                score += self.score_window(w2)

        available = [c for c in range(7) if board[0, c] == 0]
        if available:
            encoded = self.encode_state(board)
            state_tensor = torch.tensor(encoded, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor).cpu().numpy()[0]
            valid_q = [q_values[c] for c in available]
            score += float(max(valid_q)) * 2.0

        return score

    def minimax(self, board, depth, alpha, beta, is_maximizing):
        available_actions = [c for c in range(7) if board[0, c] == 0]
        ordered_actions = sorted(available_actions, key=lambda c: abs(c - 3))

        if not ordered_actions:
            return 0.0, None

        if depth == 0:
            return self.evaluate_board(board), None

        if is_maximizing:
            for a in ordered_actions:
                r = self.get_drop_row(board, a)
                board[r, a] = 1
                if self.check_line_of_four(board, r, a, 1):
                    board[r, a] = 0
                    return 10000.0 + depth * 100.0, a
                board[r, a] = 0

            max_eval = -float('inf')
            best_action = ordered_actions[0]

            for a in ordered_actions:
                r = self.get_drop_row(board, a)
                board[r, a] = 1
                eval_score, _ = self.minimax(board, depth - 1, alpha, beta, False)
                board[r, a] = 0

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = a
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_action
        else:
            for a in ordered_actions:
                r = self.get_drop_row(board, a)
                board[r, a] = -1
                if self.check_line_of_four(board, r, a, -1):
                    board[r, a] = 0
                    return -10000.0 - depth * 100.0, a
                board[r, a] = 0

            min_eval = float('inf')
            best_action = ordered_actions[0]

            for a in ordered_actions:
                r = self.get_drop_row(board, a)
                board[r, a] = -1
                eval_score, _ = self.minimax(board, depth - 1, alpha, beta, True)
                board[r, a] = 0

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = a
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_action

    def choose_action(self, state, available_actions, use_tactics=True, search_depth=5):
        if not available_actions:
            return None

        if random.random() < self.epsilon:
            return random.choice(available_actions)

        if use_tactics:
            board_copy = state.copy()
            _, best_move = self.minimax(board_copy, search_depth, -float('inf'), float('inf'), True)
            if best_move is not None and best_move in available_actions:
                return best_move

        encoded = self.encode_state(state)
        state_tensor = torch.tensor(encoded, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]

        masked_q = [(a, q_values[a]) for a in available_actions]
        max_q = max(masked_q, key=lambda x: x[1])[1]
        best_actions = [a for a, q in masked_q if q == max_q]
        return random.choice(best_actions)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states_enc = self.encode_batch(states)
        next_states_enc = self.encode_batch(next_states)

        states_t = torch.tensor(states_enc, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states_enc, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q = self.policy_net(states_t).gather(1, actions_t)

        with torch.no_grad():
            next_q = self.target_net(next_states_t)
            is_full = (torch.tensor(next_states[:, 0, :] != 0, device=self.device))
            next_q[is_full] = -1e9
            max_next_q = next_q.max(dim=1, keepdim=True)[0]
            target_q = rewards_t + (1 - dones_t) * self.gamma * max_next_q

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_steps += 1
        if self.update_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filename="dqn_connect4.pth"):
        torch.save(self.policy_net.state_dict(), filename)

    def load_model(self, filename="dqn_connect4.pth"):
        self.policy_net.load_state_dict(torch.load(filename, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())