from agent import player
from environment import blackjack
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def moving_average(data, window_size):
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')

env = blackjack()

player1 = player(env = env, lr = 1e-5, batch_size = 512, discount = 1, epsilon = 0.4, 
                 collect_period = 50, update_period = 50, decay_rate = 0.9, 
                 decay_period = 250, total_steps = 10000, use_ddqn=True)

q_values, losses = player1.train()


smoothed_q_values = moving_average(q_values, window_size=200)
smoothed_losses = moving_average(losses, window_size=200)

# Create subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 8))  # 2 rows, 1 column

# Plot smoothed Q-values
axs[0].plot(smoothed_q_values, color='blue')
axs[0].set_title('Smoothed Q-Values over Training Steps')
axs[0].set_xlabel('Steps')
axs[0].set_ylabel('Q-Value')

# Plot smoothed Losses
axs[1].plot(smoothed_losses, color='red')
axs[1].set_title('Smoothed Losses over Training Steps')
axs[1].set_xlabel('Steps')
axs[1].set_ylabel('Loss')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show plot
plt.show()

policy_win_rate = player1.test(is_random = False)
random_win_rate = player1.test(is_random = True)

print("policy win rate: {} random win rate: {}".format(policy_win_rate, random_win_rate))
