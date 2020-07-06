import numpy as np
from tf_dqn.common import utils


class ACMemory:
    """Latest memory."""

    def __init__(self, max_steps):
        self._max_steps = max_steps
        self._samples = None
        self._index = 0
        self._is_full = False
        self._indexing_array = np.arange(self._max_steps)

    def get_size(self):
        if self._is_full:
            return self._max_steps
        else:
            return self._index

    def add_sample(self, state, action, td_target):
        if self._samples is None:
            # initialize memory to correct size
            self._samples = dict(
                state={},
                action={},
                td_target=np.zeros(self._max_steps, dtype=np.float32)
            )

            # calculate experience replay memory size
            bytes_per_experience = self._samples['td_target'].itemsize

            for key in state:
                shape = tuple([self._max_steps] + list(state[key].shape))
                self._samples['state'][key] = np.zeros(shape, dtype=state[key].dtype)
                array_size = 1
                for dim in list(state[key].shape):
                    array_size *= dim
                bytes_per_experience += self._samples['state'][key].itemsize * array_size

            for key in action:
                shape = tuple([self._max_steps] + [1])
                self._samples['action'][key] = np.zeros(shape, dtype=np.int32)
                bytes_per_experience += self._samples['action'][key].itemsize

            print('Memory per experience: ' + utils.bytes_dec_to_bin(bytes_per_experience))
            total_size = utils.bytes_dec_to_bin(bytes_per_experience * self._max_steps)
            print('Total replay memory size (' + str(self._max_steps) + ' experiences): ' + total_size)

        # replace memory in index position
        for key in state:
            self._samples['state'][key][self._index, ...] = state[key]
        for key in action:
            self._samples['action'][key][self._index, ...] = action[key]

        self._samples['td_target'][self._index] = td_target

        self._index += 1
        if self._index == self._max_steps:
            self._is_full = True
        self._index = self._index % self._max_steps

    def sample(self, num_samples):
        state = {}
        for k, v in self._samples['state'].items():
            state[k] = v[:]

        action = {}
        for k, v in self._samples['action'].items():
            action[k] = v[:]

        td_target = self._samples['td_target'][:]

        self._index = 0
        self._is_full = False

        return state, action, td_target
