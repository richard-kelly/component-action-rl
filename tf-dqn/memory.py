import random


class Memory:
    """Latest replay memory."""

    def __init__(self, max_steps):
        self._max_steps = max_steps
        self._samples = []

    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_steps:
            self._samples.pop(0)

    def sample(self, num_samples):
        if num_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples))
        else:
            return random.sample(self._samples, num_samples)
