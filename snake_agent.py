from abc import ABC, abstractmethod
from collections import deque
import random

import numpy as np
from keras.models import Sequential
from keras import layers


from exploration_strategies import epsilon_greedy, upper_confidence_bound


class SnakeAgent(ABC):

    def __init__(self, actions, state_shape, loss, optimizer, gamma, use_replay=False, initial_replay_buff=None,
                 replay_buff_size=100000, use_target_dqn=False):
        self._actions = actions

        self._gamma = gamma
        self._use_replay = use_replay
        self._use_target_dqn = use_target_dqn

        self._experience_replay = deque(initial_replay_buff, maxlen=replay_buff_size) \
            if self._use_replay and initial_replay_buff is not None \
            else deque(maxlen=1)

        self._dqn = self._build_model(state_shape, loss, optimizer)
        self._target_dqn = self._build_model(state_shape, loss, optimizer) if self._use_target_dqn else None
        self.align_target_model()

    def _build_model(self, state_shape, loss, optimizer):
        model = Sequential([
            layers.Conv2D(64, (5, 5), activation='relu', input_shape=state_shape),
            layers.Conv2D(128, (5, 5), activation='relu'),
            layers.Conv2D(256, (5, 5), activation='relu'),
            layers.Flatten(),
            layers.Dense(32, activation='relu'),
            layers.Dense(len(self._actions)),
        ])
        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=['accuracy'])
        return model

    def _store(self, board, action, reward, next_board, done):
        b = np.expand_dims(np.asarray(board).astype(np.float64), axis=0) / 255
        nb = np.expand_dims(np.asarray(next_board).astype(np.float64), axis=0) / 255

        self._experience_replay.append((b, action, reward, nb, done))

    @abstractmethod
    def act(self, board, tot_steps):
        pass

    def train(self, board, action, reward, next_board, done, sample_size):
        self._store(board, action, reward, next_board, done)

        boards = [self._experience_replay[0][0]]
        actions = [self._experience_replay[0][1]]
        rewards = [self._experience_replay[0][2]]
        next_boards = [self._experience_replay[0][3]]
        dones = [self._experience_replay[0][4]]

        if self._use_replay:
            np.random.shuffle(self._experience_replay)
            buff = random.sample(self._experience_replay, sample_size)

            sep_experience_replay = list(zip(*buff))
            boards = np.squeeze(np.array(sep_experience_replay[0]), axis=1)
            actions = np.array(sep_experience_replay[1])
            rewards = np.array(sep_experience_replay[2])
            next_boards = np.squeeze(np.array(sep_experience_replay[3]), axis=1)
            dones = np.array(sep_experience_replay[4])

        q_values = self._dqn.predict(boards, verbose=0)
        next_q_values = self._target_dqn.predict(next_boards, verbose=0) if self._use_target_dqn else \
            self._dqn.predict(next_boards, verbose=0)
        for i, done in enumerate(dones):
            if done:
                q_values[i][actions[i]] = rewards[i]
            else:
                q_values[i][actions[i]] = rewards[i] + self._gamma * np.amax(next_q_values[i])

        self._dqn.fit(boards, q_values, epochs=1, verbose=0, batch_size=256)

    def align_target_model(self):
        if self._use_target_dqn:
            self._target_dqn.set_weights(self._dqn.get_weights())


class EpsilonGreedySnakeAgent(SnakeAgent):

    def __init__(self, actions, state_shape, loss, optimizer, gamma, epsilon, epsilon_decay,
                 epsilon_min, use_replay=False, initial_replay_buff=None, replay_buff_size=100000, use_target_dqn=False,
                 ):
        super().__init__(actions, state_shape, loss, optimizer, gamma, use_replay, initial_replay_buff,
                         replay_buff_size, use_target_dqn)

        self._epsilon = epsilon
        self._epsilon_decay = epsilon_decay
        self._epsilon_min = epsilon_min

    def update_epsilon(self):
        self._epsilon = max(self._epsilon_min, self._epsilon - self._epsilon_decay)

    def epsilon(self):
        return self._epsilon

    def act(self, board, tot_steps):
        return epsilon_greedy(self._dqn, self._actions, self._epsilon, board / 255)


class UCBSnakeAgent(SnakeAgent):
    def __init__(self, actions, state_shape, loss, optimizer, gamma, confidence,
                 use_replay=False, initial_replay_buff=None, replay_buff_size=100000, use_target_dqn=False,
                 ):
        super().__init__(actions, state_shape, loss, optimizer, gamma, use_replay, initial_replay_buff,
                         replay_buff_size, use_target_dqn)

        self._confidence = confidence
        self._num_times_actions = [0, 0, 0]

    def _store(self, board, action, reward, next_board, done):
        super()._store(board, action, reward, next_board, done)
        self._num_times_actions[action+1] += 1

    def act(self, board, tot_steps):
        return upper_confidence_bound(self._dqn, self._actions, self._confidence, board / 255,
                                      tot_steps, self._num_times_actions)
