import random
import numpy as np

class Memory:
    """Latest replay memory."""

    def __init__(self, max_steps):
        self._max_steps = max_steps
        self._samples = None
        self._index = 0
        self._is_full = False

    def add_sample(self, state, action, reward, next_state, is_terminal):
        if self._samples is None:
            # initialize memory to correct size
            self._samples = dict(
                state={},
                action={},
                next_state={},
                reward=np.zeros(self._max_steps, dtype=np.float),
                is_terminal=np.full(self._max_steps, False)
            )

            for key in state.keys():
                shape = tuple([self._max_steps] + list(state[key].shape))
                self._samples['state'][key] = np.zeros(shape, dtype=np.float)
                self._samples['next_state'][key] = np.zeros(shape, dtype=np.float)

            for key in action.keys():
                shape = tuple([self._max_steps] + list(action[key].shape))
                self._samples['action'][key] = np.zeros(shape, dtype=np.float)

        # replace memory in index position
        for key in state.keys():
            self._samples['state'][key][self._index, ...] = state[key]
        for key in action.keys():
            self._samples['action'][key][self._index, ...] = action[key]
        # next_state is None for terminal states
        if next_state is not None:
            for key in next_state.keys():
                self._samples['next_state'][key][self._index, ...] = next_state[key]
        self._samples['reward'][self._index] = reward
        self._samples['is_terminal'][self._index] = is_terminal

        self._index += 1
        if self._index == self._max_steps:
            self._is_full = True
        self._index = self._index % self._max_steps

    def sample(self, num_samples):
        if self._is_full:
            if num_samples > self._max_steps:
                num_samples = self._max_steps
            indices = np.random.choice(np.arange(self._max_steps), num_samples, replace=False)
        else:
            if num_samples > self._index:
                num_samples = self._index
            indices = np.random.choice(np.arange(self._index), num_samples, replace=False)

        state = {}
        next_state = {}
        for key in self._samples['state'].keys():
            state[key] = self._samples['state'][key][indices, ...]
            next_state[key] = self._samples['next_state'][key][indices, ...]
        action = {}
        for key in self._samples['action'].keys():
            action[key] = self._samples['action'][key][indices, ...]
        reward = self._samples['reward'][indices]
        is_terminal = self._samples['is_terminal'][indices]

        return state, action, reward, next_state, is_terminal
