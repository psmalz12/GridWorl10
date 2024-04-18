from env import Grid, Goal
from visual import GridVisual
from RL_agent import RL
import random

if __name__ == "__main__":
    grid = Grid(state=(1, 1))  # the start location
    print(f"//////// agent current Location {grid.state} and represented  by the symbol: 2 ////////")
    GridVisual.show_board(grid)

    # Initialize RL agent
    epsilon = 0.8  # epsilon value high means more exploration, low means exploit from previous knowledge
    learning_rate = 0.1  # learning rate value low means the agent updates 10% of the Q-value
    rl_agent = RL(epsilon, learning_rate)


    num_iterations = 10  # Specify the number of iterations
    for iteration in range(num_iterations):
        # interact with the env(grid) until the goal is reached
        while grid.state != grid.goal:
            # Choose action using RL agent and take action in the environment
            action = rl_agent.choose_action(grid.state)
            action_state_pairs, _ = grid.action(action)

        print("Goal Reached in this iteration!")
        print(f"Total Iteration: {iteration + 1}")

        # print the cumulative reward value for each state-action pair
        for state, rewards in grid.rewards.items():
            print(f"State: {state}")
            for action, reward in rewards.items():
                print(f"  Action: {action}, Reward: {reward :.4g}")

        # Reset the agent's location for the next iteration
        grid.reset()

    print("All iterations completed.")
