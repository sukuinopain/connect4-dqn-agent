import random
import numpy as np
import matplotlib.pyplot as plt
from connect_four_env import ConnectFour
from dqn_agent import DQNAgent

def compute_positional_reward(board, action):
    center_rewards = {3: 0.12, 2: 0.06, 4: 0.06, 1: 0.02, 5: 0.02, 0: -0.04, 6: -0.04}
    reward = center_rewards[action]

    for r in range(6):
        if board[r, action] != 0:
            if r < 5 and board[r + 1, action] == -1:
                reward += 0.05
            break
    return reward

def train_dqn(episodes=2000):
    env = ConnectFour()
    agent = DQNAgent()

    win_history = []
    agent_wins = 0
    opp_wins = 0
    draws = 0
    total_steps = 0

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        agent_player = random.choice([1, -1])
        prev_state = None
        prev_action = None
        prev_shaping = 0.0

        while not done:
            available_actions = env.get_available_actions()
            current_player = env.current_player

            if current_player == agent_player:
                norm_state = state * agent_player
                action = agent.choose_action(norm_state, available_actions, use_tactics=True)
                next_state, _, done, winner = env.step(action)

                norm_next = next_state * agent_player
                shaping = compute_positional_reward(norm_next, action)

                if prev_state is not None:
                    agent.memory.push(prev_state, prev_action, prev_shaping, norm_state, False)
                    total_steps += 1
                    if total_steps % 4 == 0:
                        agent.train_step()

                prev_state = norm_state.copy()
                prev_action = action
                prev_shaping = shaping

                if done:
                    term_state = next_state * agent_player
                    if winner == agent_player:
                        agent.memory.push(prev_state, prev_action, 1.0 + shaping, term_state, True)
                        agent_wins += 1
                    elif winner == 0:
                        agent.memory.push(prev_state, prev_action, 0.3, term_state, True)
                        draws += 1
                    agent.train_step()

                state = next_state

            else:
                opp_norm_state = state * (-agent_player)
                if random.random() < 0.35:
                    center_weights = [1, 2, 4, 8, 4, 2, 1]
                    weights = [center_weights[a] for a in available_actions]
                    action = random.choices(available_actions, weights=weights, k=1)[0]
                else:
                    old_eps = agent.epsilon
                    agent.epsilon = 0.15
                    action = agent.choose_action(opp_norm_state, available_actions, use_tactics=True)
                    agent.epsilon = old_eps

                next_state, _, done, winner = env.step(action)

                if done:
                    term_state = next_state * agent_player
                    if winner == -agent_player:
                        if prev_state is not None:
                            agent.memory.push(prev_state, prev_action, -1.0, term_state, True)
                        opp_wins += 1
                    elif winner == 0:
                        if prev_state is not None:
                            agent.memory.push(prev_state, prev_action, 0.3, term_state, True)
                        draws += 1
                    agent.train_step()

                state = next_state

        agent.decay_epsilon()

        if episode % 100 == 0:
            total = agent_wins + opp_wins + draws
            win_rate = agent_wins / total
            loss_rate = opp_wins / total
            draw_rate = draws / total
            win_history.append((win_rate, loss_rate, draw_rate))

            print(f"Episode {episode}/{episodes} | Win: {win_rate*100:.1f}% | Loss: {loss_rate*100:.1f}% | Draw: {draw_rate*100:.1f}% | Epsilon: {agent.epsilon:.3f}")
            agent_wins, opp_wins, draws = 0, 0, 0

    agent.save_model("dqn_connect4.pth")
    print("Training complete. Model saved to dqn_connect4.pth")
    return win_history

if __name__ == "__main__":
    train_dqn(episodes=2000)