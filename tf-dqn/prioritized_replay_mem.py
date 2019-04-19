import math
import numpy as np
from latest_replay_mem import LatestReplayMemory


class PrioritizedReplayMemory(LatestReplayMemory):
    """Prioritized Experience Replay Memory."""

    def __init__(self, max_steps, alpha, beta, steps_to_anneal_b_to_1):
        super().__init__(max_steps)
        self._alpha = alpha
        self._beta = beta
        self._beta_change = (1.0 - beta) / steps_to_anneal_b_to_1

        self._max_priority = 1.0

        self._last_indices = None

        # initialize sum tree
        # round up to nearest power of 2 for tree size
        power = 1
        while power <= self._max_steps:
            power *= 2

        self._tree_offset = power - 1
        self._tree = np.zeros(power * 2 - 1, dtype=np.float32)

    def _set_priority(self, index, priority):
        # initial index of priority in tree
        tree_index = self._tree_offset + index

        # difference to propogate up the tree
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
                priority -= left
                tree_index = left + 1

        # return adjusted index of leaf node
        return tree_index - self._tree_offset

    def add_sample(self, state, action, reward, next_state, is_terminal):
        # update beta
        self._beta += self._beta_change

        # set priority of new transition to max priority
        self._set_priority(self._index, self._max_priority ** self._alpha)

        # add sample (this increments self._index)
        super().add_sample(state, action, reward, next_state, is_terminal)

    def _get_sample_indices(self, num_samples):
        # TODO: implement


    def update_priorities_of_last_sample(self, new_priorities):
        # TODO: implement

    def sample(self, num_samples):
        indices = self._get_sample_indices(num_samples)
        self._last_indices = indices

        state, action, reward, next_state, is_terminal = self._get_transitions(indices)

        # TODO: calculate weights and add those to returned tuple
        weights = None

        return state, action, reward, next_state, is_terminal, weights