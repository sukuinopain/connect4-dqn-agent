import random
import torch
import numpy as np
from connect_four_env import ConnectFour
from dqn_agent import DQNAgent

def evaluate_agent(games=100):
    env = ConnectFour()
    agent = DQNAgent(epsilon=0.0)
    agent.load_model("dqn_connect4.pth")

    print(f"\n--- Evaluating DQN Agent as First Player (X) vs Random ({games} games) ---")
    agent_wins = 0
    random_wins = 0
    draws = 0

    for _ in range(games):
        state = env.reset()
        done = False
        while not done:
            available_actions = env.get_available_actions()
            if env.current_player == 1:
                norm_state = state * 1
                action = agent.choose_action(norm_state, available_actions)
            else:
                action = random.choice(available_actions)

            state, reward, done, winner = env.step(action)

        if winner == 1:
            agent_wins += 1
        elif winner == -1:
            random_wins += 1
        else:
            draws += 1

    print(f"Agent Win Rate:   {agent_wins / games * 100:.2f}% ({agent_wins}/{games})")
    print(f"Random Opponent:  {random_wins / games * 100:.2f}% ({random_wins}/{games})")
    print(f"Draw Rate:        {draws / games * 100:.2f}% ({draws}/{games})")

    print(f"\n--- Evaluating DQN Agent as Second Player (O) vs Random ({games} games) ---")
    agent_wins_2 = 0
    random_wins_2 = 0
    draws_2 = 0

    for _ in range(games):
        state = env.reset()
        done = False
        while not done:
            available_actions = env.get_available_actions()
            if env.current_player == 1:
                action = random.choice(available_actions)
            else:
                norm_state = state * -1
                action = agent.choose_action(norm_state, available_actions)

            state, reward, done, winner = env.step(action)

        if winner == -1:
            agent_wins_2 += 1
        elif winner == 1:
            random_wins_2 += 1
        else:
            draws_2 += 1

    print(f"Agent Win Rate:   {agent_wins_2 / games * 100:.2f}% ({agent_wins_2}/{games})")
    print(f"Random Opponent:  {random_wins_2 / games * 100:.2f}% ({random_wins_2}/{games})")
    print(f"Draw Rate:        {draws_2 / games * 100:.2f}% ({draws_2}/{games})")

if __name__ == "__main__":
    evaluate_agent(games=100)