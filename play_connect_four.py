import os
from connect_four_env import ConnectFour
from dqn_agent import DQNAgent

def play_game():
    if not os.path.exists("dqn_connect4.pth"):
        print("Model file 'dqn_connect4.pth' not found! Train the agent first.")
        return

    agent = DQNAgent(epsilon=0.0)
    agent.load_model("dqn_connect4.pth")
    env = ConnectFour()

    print("--- Connect Four: Human vs Enhanced Tactical DQN ---")
    print("Choose your turn:")
    print("1: Play First (X)")
    print("2: Play Second (O)")

    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice in ["1", "2"]:
            break
        print("Invalid choice! Please enter 1 or 2.")

    human_player = 1 if choice == "1" else -1
    agent_player = -human_player

    state = env.reset()
    done = False

    while not done:
        env.render()
        available_actions = env.get_available_actions()

        if env.current_player == human_player:
            valid_move = False
            while not valid_move:
                try:
                    col = int(input(f"Your turn ({'X' if human_player == 1 else 'O'}). Enter column (0-6): "))
                    if col in available_actions:
                        valid_move = True
                        state, reward, done, winner = env.step(col)
                    else:
                        print(f"Column {col} is full or invalid! Pick from: {available_actions}")
                except ValueError:
                    print("Please enter a valid number between 0 and 6.")
        else:
            print("Agent is thinking...")
            norm_state = state * agent_player
            action = agent.choose_action(norm_state, available_actions, use_tactics=True)
            print(f"Agent played column: {action}")
            state, reward, done, winner = env.step(action)

    print("\nFinal Board:")
    env.render()

    if winner == human_player:
        print("Congratulations! You beat the DQN Agent!")
    elif winner == agent_player:
        print("Agent won! Better luck next time.")
    else:
        print("It's a draw!")

if __name__ == "__main__":
    play_game()