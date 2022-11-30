import numpy as np

def epsilon_greedy(q, epsilon):
    """Epsilon-greedy policy: selects the maximum value action with probabilty
    (1-epsilon) and selects randomly with epsilon probability.

    Args:
      q (ndarray): an array of action values
      epsilon (float): probability of selecting an action randomly

    Returns:
      int: the chosen action
    """
    be_greedy = np.random.random() > epsilon
    if be_greedy:
        action = np.argmax(q)
    else:
        action = np.random.choice(len(q))

    return action


def q_learning(state, action, reward, next_state, value, params):
    """Q-learning: updates the value function and returns it.

    Args:
    state (int): the current state identifier
    action (int): the action taken
    reward (float): the reward received
    next_state (int): the transitioned to state identifier
    value (ndarray): current value function of shape (n_states, n_actions)
    params (dict): a dictionary containing the default parameters

    Returns:
    ndarray: the updated value function of shape (n_states, n_actions)
    """
    # value of previous state-action pair
    prev_value = value[int(state), int(action)]

    # maximum Q-value at current state
    if next_state is None or np.isnan(next_state):
        max_value = 0
    else:
        max_value = np.max(value[int(next_state)])

    # reward prediction error
    delta = reward + params['gamma'] * max_value - prev_value

    # update value of previous state-action pair
    value[int(state), int(action)] = prev_value + params['alpha'] * delta

    return value

def learn_environment(env, model_updater, planner, params, max_steps, n_episodes, shortcut_episode=None):
    # Start with a uniform value function
    value = np.ones((env.n_states, env.n_actions))

    # Run learning
    reward_sums = np.zeros(n_episodes)
    episode_steps = np.zeros(n_episodes)

    # Dyna-Q state
    model = np.nan*np.zeros((env.n_states, env.n_actions, 2))

    # Loop over episodes
    for episode in range(n_episodes):
        if shortcut_episode is not None and episode == shortcut_episode:
            env.toggle_shortcut()
            state = 64
            action = 1
            next_state, reward = env.get_outcome(state, action)
            model[state, action] = reward, next_state
            value = q_learning(state, action, reward, next_state, value, params)


        state = env.init_state  # initialize state
        reward_sum = 0

        for t in range(max_steps):
            # choose next action
            action = epsilon_greedy(value[state], params['epsilon'])

            # observe outcome of action on environment
            next_state, reward = env.get_outcome(state, action)

            # sum rewards obtained
            reward_sum += reward

            # update value function
            value = q_learning(state, action, reward, next_state, value, params)

            # update model
            model = model_updater(model, state, action, reward, next_state)

            # execute planner
            value = planner(model, value, params)

            if next_state is None:
                break  # episode ends
            state = next_state

        reward_sums[episode] = reward_sum
        episode_steps[episode] = t+1

    return value, reward_sums, episode_steps
