import math
import random
import numpy as np
from latest_replay_mem import LatestReplayMemory


class PrioritizedReplayMemory(LatestReplayMemory):
    """Prioritized Experience Replay Memory."""

    def __init__(self, max_steps, alpha, beta, steps_to_anneal_b_to_1):
        super().__init__(max_steps)
        self._alpha = alpha
        self._beta = beta
        self._beta_change = (1.0 - beta) / steps_to_anneal_b_to_1
        self._epsilon = 1e-6
        self._max_priority = 1.0

        self._last_indices = None

        # initialize sum tree
        # round up to nearest power of 2 for tree size
        power = 1
        while power < self._max_steps:
            power *= 2

        self._tree_offset = power - 1
        self._tree = np.zeros(power * 2 - 1, dtype=np.float64)

    def _set_priority(self, index, priority):
        # initial index of priority in tree
        tree_index = self._tree_offset + index

        # apply prioritization parameter
        priority = priority ** self._alpha

        # difference to propagate up the tree
        d = priority - self._tree[tree_index]

        # update the leaf
        self._tree[tree_index] = priority

        # update inner nodes and root
        while tree_index != 0:
            tree_index = math.ceil(tree_index / 2 - 1)
            self._tree[tree_index] += d

    def _get_index(self, priority):
        # descend tree from root
        tree_index = 0
        while tree_index < self._tree_offset:
            left = tree_index * 2 + 1
            if priority <= self._tree[left]:
                tree_index = left
            else:
                priority -= self._tree[left]
                tree_index = left + 1
                # precision error can cause bad situation where priority is higher than new node
                priority = min(priority, self._tree[tree_index])
        # return adjusted index of leaf node
        return tree_index - self._tree_offset

    def add_sample(self, state, action, reward, next_state, is_terminal):
        # update beta
        self._beta += self._beta_change

        # set priority of new transition to max priority
        self._set_priority(self._index, self._max_priority)

        # add sample (this increments self._index)
        super().add_sample(state, action, reward, next_state, is_terminal)

    def _get_sample_indices(self, num_samples):
        indices = np.zeros(num_samples, dtype=np.int32)
        size = self._tree[0] / num_samples
        self.sample_priorities = np.zeros(num_samples, dtype=np.float32)
        for i in range(num_samples):
            segment = size * i
            val = random.uniform(segment, segment + size)
            self.sample_priorities[i] = val
            indices[i] = self._get_index(val)
        return indices

    def update_priorities_of_last_sample(self, new_priorities):
        # update max_priority
        self._max_priority = max(self._max_priority, new_priorities.max())

        # add epsilon to ensure priorities are non-zero
        new_priorities += self._epsilon
        for i in range(new_priorities.shape[0]):
            self._set_priority(self._last_indices[i], new_priorities[i])

    def sample(self, num_samples):
        indices = self._get_sample_indices(num_samples)
        # store indices so we can update their priorities after training on this batch
        self._last_indices = indices

        state, action, reward, next_state, is_terminal = self._get_transitions(indices)

        priorities = self._tree[indices + self._tree_offset]
        probabilities = priorities / self._tree[0]
        weights = (self.get_size() * probabilities) ** -self._beta
        weights /= weights.max()

        return state, action, reward, next_state, is_terminal, weights
