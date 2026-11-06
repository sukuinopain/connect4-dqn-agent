# Connect Four AI - Tactical DQN Agent

A smart Connect Four AI agent that combines Deep Q-Networks (DQN) with an Alpha-Beta search engine to play at an expert level.

---

## Benchmark Results

The agent was tested across multiple scenarios:

- **vs Random Opponent (Player 1):** 100% Win Rate (100/100 games)
- **vs Random Opponent (Player 2):** 100% Win Rate (100/100 games)
- **vs Heuristic Minimax Engine:** 100% Win Rate (40/40 games)
- **vs Human Players:** High tactical performance with double-threat setups

---

## How It Works

- **Convolutional Neural Network (CNN):** Instead of using a simple table, the AI looks at the 6x7 board using 2D convolutions to recognize patterns, lines, and diagonals.
- **Two-Channel Representation:** The board is split into two separate views (one for the AI's pieces and one for the opponent) so the network clearly sees both sides.
- **Tactical Lookahead:** Uses a 5-step lookahead search to block opponent wins and secure quick victories before making a move.

---

## Project Files

- `connect_four_env.py` : Game logic and rules for Connect Four.
- `dqn_agent.py` : The AI model, replay buffer, and search algorithm.
- `train_dqn.py` : Training script for the neural network.
- `evaluate_dqn.py` : Evaluation script against random play.
- `eval_vs_minimax.py` : Benchmark test against a Minimax engine.
- `gui_connect_four.py` : Playable graphical interface using Pygame.
- `play_connect_four.py` : Playable console version inside the terminal.
- `dqn_connect4.pth` : Saved weights of the trained neural network.

## Training Performance

![Training Convergence](training_performance.png)