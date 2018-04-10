# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["ConstantStepSize", "VariableStepSize"]

import numpy as np


class ConstantStepSize(object):

    def __init__(self, step_size, jitter=None):
        self.step_size = step_size
        self.jitter = jitter

    def sample_step_size(self, random=None):
        if random is None:
            random = np.random
        jitter = self.jitter
        eps = self.get_step_size()
        if jitter is None:
            return eps
        jitter = np.clip(jitter, 0, 1)
        return eps * (1.0 - jitter * random.uniform(-1, 1))

    def get_step_size(self):
        return self.step_size

    def restart(self):
        pass

    def update(self, accept_stat):
        pass

    def finalize(self):
        pass


class VariableStepSize(ConstantStepSize):

    def __init__(self, delta=0.5, mu=0.5, gamma=0.05, kappa=0.75, t0=10,
                 jitter=None):
        self.jitter = jitter
        self.mu = mu
        self.delta = delta
        self.gamma = gamma
        self.kappa = kappa
        self.t0 = t0
        self.restart()

    def restart(self):
        self.counter = 0
        self.s_bar = 0.0
        self.x_bar = 0.0
        self.x = 0.0

    def get_step_size(self):
        return np.exp(self.x)

    def update(self, adapt_stat):
        self.counter += 1
        adapt_stat = min(adapt_stat, 1.0)
        eta = 1.0 / (self.counter + self.t0)
        self.s_bar = (1.0 - eta) * self.s_bar + eta * (self.delta - adapt_stat)
        self.x = self.mu - self.s_bar * np.sqrt(self.counter) / self.gamma
        x_eta = self.counter ** -self.kappa
        self.x_bar = (1.0 - x_eta) * self.x_bar + x_eta * self.x

    def finalize(self):
        self.x = self.x_bar
