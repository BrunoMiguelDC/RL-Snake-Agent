import math

import numpy as np


def epsilon_greedy(dqn, actions, epsilon, board):
    if np.random.rand() <= epsilon:
        return np.random.choice(actions)

    board = np.expand_dims(np.asarray(board).astype(np.float32), axis=0)

    q_values = dqn.predict(board, verbose=0)
    max_index = np.argmax(q_values[0])
    return actions[max_index]


def upper_confidence_bound(dqn, actions, confidence, board, tot_steps, num_times_actions):
    board = np.expand_dims(np.asarray(board).astype(np.float32), axis=0)
    upper_bounds = []
    q_values = dqn.predict(board, verbose=0)
    for i in range(len(q_values)):
        if num_times_actions[i] > 0:
            upper_bound = q_values[i] + confidence * math.sqrt(math.log(tot_steps) / num_times_actions[i])
        else:
            upper_bound = 10e500  # very large value
        upper_bounds.append(upper_bound)
    max_index = np.argmax(upper_bounds)
    return actions[max_index]
