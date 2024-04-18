import random

class RL:
    def __init__(self, epsilon, learning_rate):
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.q_values = {}

    def choose_action(self, state):
        if random.random() < self.epsilon or state not in self.q_values:
            # randomly select action if exploring (less than epsilon) or if the state has no Q-values
            return random.choice([0, 1, 2, 3])  # (0: up, 1: down, 2: right, 3: left)
        else:
            # otherwise get the action with the highest Q-value
            max_actions = [action for action in self.q_values[state] if self.q_values[state][action] == max(self.q_values[state].values())]
            # randomly choose one of the actions with the highest q-value (this will benifit if we have 2 action have the same q-value)
            return random.choice(max_actions)

    def update_q_values(self, state, action, reward, next_state):
        # update Q-values using the update rule
        if state not in self.q_values or action not in self.q_values[state]:self.q_values[state][action] = reward

        else:
            # get the maximum expected future reward
            max_next_reward = max(self.q_values.get(next_state, {}).values())
            self.q_values[state][action] += self.learning_rate * (reward + max_next_reward - self.q_values[state][action])
            # Q(s, a) = Q(s, a) + alpha * (reward + max(Q(s', a')) - Q(s, a))