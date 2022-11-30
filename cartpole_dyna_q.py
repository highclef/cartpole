import gym
import numpy as np
import matplotlib.pyplot as plt

from helper import learn_environment
from dyna_q import dyna_q_model_update, dyna_q_planning
from plotter import plot_performance

ENV_NAME = "CartPole-v1"

# set for reproducibility, comment out / change seed value for different results
np.random.seed(1)

# parameters needed by our policy and learning rule
params = {
    'epsilon': 0.05,  # epsilon-greedy policy
    'alpha': 0.5,  # learning rate
    'gamma': 0.8,  # temporal discount factor
    'k': 10,  # number of Dyna-Q planning steps
}

# episodes/trials
n_episodes = 500
max_steps = 1000

# environment initialization
env = gym.make(ENV_NAME)

# solve Quentin's World using Dyna-Q
value, reward_sums, episode_steps = learn_environment(env, dyna_q_model_update, dyna_q_planning, params, max_steps, n_episodes)

# Plot the results
with plt.xkcd():
    plot_performance(env, value, reward_sums, n_episodes)