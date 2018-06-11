# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["NoUTurnSampler"]

import logging
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import time

from tqdm import tqdm

from .integrator import leapfrog
from .metric import IdentityMetric
from .step_size import VariableStepSize, ConstantStepSize


Point = namedtuple("Point", ("q", "p", "U", "dUdq"))


def _nuts_criterion(p_sharp_minus, p_sharp_plus, rho):
    return np.dot(p_sharp_plus, rho) > 0 and np.dot(p_sharp_minus, rho) > 0


def _nuts_tree(log_prob_fn, grad_log_prob_fn, metric, epsilon,
               depth, z, z_propose, p_sharp_left, p_sharp_right, rho, H0,
               sign, n_leapfrog, log_sum_weight, sum_metro_prob, max_depth,
               max_delta_h, random):
    if depth == 0:
        q, p, dUdq = leapfrog(grad_log_prob_fn, metric, z.q, z.p,
                              sign * epsilon, z.dUdq)
        z = Point(q, p, -log_prob_fn(q), dUdq)
        n_leapfrog += 1

        h = 0.5 * np.dot(p, metric.dot(p))
        h += z.U
        if not np.isfinite(h):
            h = np.inf
        valid_subtree = (h - H0) <= max_delta_h

        log_sum_weight = np.logaddexp(log_sum_weight, H0 - h)
        sum_metro_prob += min(np.exp(H0 - h), 1.0)

        z_propose = z
        rho += z.p

        p_sharp_left = metric.dot(z.p)
        p_sharp_right = p_sharp_left

        return (
            valid_subtree, z, z_propose, p_sharp_left, p_sharp_right, rho,
            n_leapfrog, log_sum_weight, sum_metro_prob
        )

    p_sharp_dummy = np.empty_like(p_sharp_left)

    # Left
    log_sum_weight_left = -np.inf
    rho_left = np.zeros_like(rho)
    results_left = _nuts_tree(
        log_prob_fn, grad_log_prob_fn, metric, epsilon,
        depth - 1, z, z_propose, p_sharp_left, p_sharp_dummy, rho_left,
        H0, sign, n_leapfrog, log_sum_weight_left, sum_metro_prob, max_depth,
        max_delta_h, random
    )
    (valid_left, z, z_propose, p_sharp_left, p_sharp_dummy, rho_left,
     n_leapfrog, log_sum_weight_left, sum_metro_prob) = results_left

    if not valid_left:
        return (
            False, z, z_propose, p_sharp_left, p_sharp_right, rho,
            n_leapfrog, log_sum_weight, sum_metro_prob
        )

    # Right
    z_propose_right = Point(z.q, z.p, z.U, z.dUdq)
    log_sum_weight_right = -np.inf
    rho_right = np.zeros_like(rho)
    results_right = _nuts_tree(
        log_prob_fn, grad_log_prob_fn, metric, epsilon,
        depth - 1, z, z_propose_right, p_sharp_dummy, p_sharp_right, rho_right,
        H0, sign, n_leapfrog, log_sum_weight_right, sum_metro_prob, max_depth,
        max_delta_h, random
    )
    (valid_right, z, z_propose_right, p_sharp_dummy, p_sharp_right, rho_right,
     n_leapfrog, log_sum_weight_right, sum_metro_prob) = results_right

    if not valid_right:
        return (
            False, z, z_propose, p_sharp_left, p_sharp_right, rho,
            n_leapfrog, log_sum_weight, sum_metro_prob
        )

    # Multinomial sample from the right
    log_sum_weight_subtree = np.logaddexp(log_sum_weight_left,
                                          log_sum_weight_right)
    log_sum_weight = np.logaddexp(log_sum_weight, log_sum_weight_subtree)

    if log_sum_weight_right > log_sum_weight_subtree:
        z_propose = z_propose_right
    else:
        accept_prob = np.exp(log_sum_weight_right - log_sum_weight_subtree)
        if random.rand() < accept_prob:
            z_propose = z_propose_right

    rho_subtree = rho_left + rho_right
    rho += rho_subtree

    return (
        _nuts_criterion(p_sharp_left, p_sharp_right, rho_subtree),
        z, z_propose, p_sharp_left, p_sharp_right, rho,
        n_leapfrog, log_sum_weight, sum_metro_prob
    )


def step_nuts(log_prob_fn, grad_log_prob_fn, metric, q, log_prob, epsilon,
              max_depth, max_delta_h, random):
    dUdq = -grad_log_prob_fn(q)
    p = metric.sample_p(random=random)

    z_plus = Point(q, p, -log_prob, dUdq)
    z_minus = Point(q, p, -log_prob, dUdq)
    z_sample = Point(q, p, -log_prob, dUdq)
    z_propose = Point(q, p, -log_prob, dUdq)

    p_sharp_plus = metric.dot(p)
    p_sharp_dummy = np.array(p_sharp_plus, copy=True)
    p_sharp_minus = np.array(p_sharp_plus, copy=True)
    rho = np.array(p, copy=True)

    n_leapfrog = 0
    log_sum_weight = 0.0
    sum_metro_prob = 0.0
    H0 = 0.5 * np.dot(p, metric.dot(p))
    H0 -= log_prob

    for depth in range(max_depth):
        rho_subtree = np.zeros_like(rho)
        valid_subtree = False
        log_sum_weight_subtree = -np.inf

        if random.rand() > 0.5:
            results = _nuts_tree(
                log_prob_fn, grad_log_prob_fn, metric, epsilon,
                depth, z_plus, z_propose, p_sharp_dummy, p_sharp_plus,
                rho_subtree, H0, 1, n_leapfrog, log_sum_weight_subtree,
                sum_metro_prob, max_depth, max_delta_h, random)
            (valid_subtree, z_plus, z_propose, p_sharp_dummy, p_sharp_plus,
             rho_subtree, n_leapfrog, log_sum_weight_subtree, sum_metro_prob) \
                = results

        else:
            results = _nuts_tree(
                log_prob_fn, grad_log_prob_fn, metric, epsilon,
                depth, z_minus, z_propose, p_sharp_dummy, p_sharp_minus,
                rho_subtree, H0, -1, n_leapfrog, log_sum_weight_subtree,
                sum_metro_prob, max_depth, max_delta_h, random)
            (valid_subtree, z_minus, z_propose, p_sharp_dummy, p_sharp_minus,
             rho_subtree, n_leapfrog, log_sum_weight_subtree, sum_metro_prob) \
                = results

        if not valid_subtree:
            break

        if log_sum_weight_subtree > log_sum_weight:
            z_sample = z_propose
        else:
            accept_prob = np.exp(log_sum_weight_subtree - log_sum_weight)
            if random.rand() < accept_prob:
                z_sample = z_propose

        log_sum_weight = np.logaddexp(log_sum_weight, log_sum_weight_subtree)
        rho += rho_subtree

        if not _nuts_criterion(p_sharp_minus, p_sharp_plus, rho):
            break

    accept_prob = sum_metro_prob / n_leapfrog
    return z_sample.q, log_prob_fn(q), float(accept_prob), z_sample.p


class NoUTurnSampler(object):

    def __init__(self, log_prob_fn, grad_log_prob_fn,
                 step_size=None, metric=None,
                 max_depth=5, max_delta_h=1000.0):
        self.log_prob_fn = log_prob_fn
        self.grad_log_prob_fn = grad_log_prob_fn
        self.step_size = step_size
        self.metric = metric
        self.max_depth = max_depth
        self.max_delta_h = max_delta_h

    def sample(self, initial_q, n_sample, initial_log_prob=None,
               tune=False, tune_step_size=False, tune_metric=False,
               random=None, title=None, top_pbar=None):
        if random is None:
            random = np.random
        if title is None:
            title = ""

        # Set up the default step size estimator
        if self.step_size is None:
            self.step_size = VariableStepSize()

        # If the provided step size was a number, assume a constant step size
        try:
            self.step_size.sample_step_size(random=random)
        except AttributeError:
            self.step_size = ConstantStepSize(self.step_size)

        # Set up the default metric
        if self.metric is None:
            self.metric = IdentityMetric(len(initial_q))
        if len(initial_q) != self.metric.ndim:
            raise ValueError("dimension mismatch between initial coordinates "
                             "and metric")

        # Compute the initial log probability if needed
        if initial_log_prob is None:
            initial_log_prob = self.log_prob_fn(initial_q)
        log_prob = initial_log_prob
        q = initial_q

        # Stats that we will accumulate during this run
        accept_stat = 0

        # Do the sampling
        leave = top_pbar is None
        with tqdm(range(n_sample), total=n_sample, leave=leave) as pbar:
            for n in pbar:
                # Sample the step size including jitter
                step = self.step_size.sample_step_size(random=random)

                # Run one step of NUTS
                q, log_prob, accept, p = step_nuts(
                    self.log_prob_fn, self.grad_log_prob_fn, self.metric,
                    q, log_prob, step, self.max_depth, self.max_delta_h,
                    random)

                # Update the stats
                accept_stat += accept

                # Optionally tune the step size and metric
                if tune or tune_step_size:
                    self.step_size.update(accept)
                if tune or tune_metric:
                    self.metric.update(q)

                # Update the stats in the progress bar
                pbar.set_description(
                    title + "step_size: {0:.1e}; mean(accept_stat): {1:.3f}"
                    .format(step, accept_stat/(n+1)))

                # Update the top level progress bar
                if top_pbar is not None:
                    top_pbar.update(1)

                # Yield the current state
                yield q, log_prob, p

        # If tuning, finalize the step size and metric. If the metric is
        # updated, the step size should be restarted so that it can be learned
        # again.
        if tune or tune_step_size:
            self.step_size.finalize()
        if tune or tune_metric:
            self.metric.finalize()
            self.step_size.restart()

    def run_warmup(self, initial_q, n_warmup, initial_log_prob=None,
                   tune_step_size=True, tune_metric=True,
                   initial_buffer=100, final_buffer=100, window=25,
                   random=None, plot=False, update_interval=50):

        if plot:
            fig_warmup, ax_warmup = plt.subplots(2, len(initial_q))
            ax_warmup = list(map(list, zip(*ax_warmup)))   # I disgust myself

            chain_q = np.empty((n_warmup, len(initial_q)), dtype=float)
            chain_p = np.empty((n_warmup, len(initial_q)), dtype=float)
        
        # Compute the schedule for the warm up
        if tune_metric:
            # First, how many inner steps do we get?
            n_inner = n_warmup - initial_buffer - final_buffer
            if n_inner <= window:
                logging.warn("not enough warm up samples for proposed "
                             "schedule; resizing to 20%/70%/10%")
                initial_buffer, n_inner, final_buffer = \
                    (np.array([0.2, 0.7, 0.1]) * n_warmup).astype(int)

            # Compute the tuning schedule
            p = max(1, np.ceil(np.log2(n_inner) - np.log2(window)) + 1)
            windows = window * 2 ** np.arange(p)
            if len(windows) <= 1:
                windows = np.array([n_inner])
            else:
                if windows[-1] > n_inner:
                    windows = np.append(windows[:-2], n_inner)
            windows = np.diff(np.append(0, windows)).astype(int)
        else:
            initial_buffer = 0
            final_buffer = n_warmup
            windows = []

        # Run a first pass where only the step size is tuned
        q = initial_q
        log_prob = initial_log_prob

        # with tqdm(range(n_warmup), total=n_warmup) as pbar:
        # pbar.set_description("warm up: ")
        pbar = None
        if initial_buffer > 0:
            for q, log_prob, p in self.sample(q, initial_buffer,
                                           initial_log_prob=log_prob,
                                           tune_step_size=tune_step_size,
                                           random=random,
                                           top_pbar=pbar,
                                           title="initial warm up: "):
                pass

        # For each window in the schedule tune the metric
        for n, w in enumerate(windows):
            for q, log_prob, p in self.sample(q, w,
                                           initial_log_prob=log_prob,
                                           tune_step_size=tune_step_size,
                                           tune_metric=tune_metric,
                                           random=random,
                                           top_pbar=pbar,
                                           title="warm up {0}/{1}: "
                                           .format(n+1, len(windows))):
                pass

        # Using the final metric, tune the step size
        if final_buffer > 0:
            for q, log_prob, p in self.sample(q, final_buffer,
                                           initial_log_prob=log_prob,
                                           tune_step_size=tune_step_size,
                                           random=random,
                                           top_pbar=pbar,
                                           title="final warm up: "):
                pass

        return q, log_prob

    def run_mcmc(self, initial_q, n_mcmc, tune=False, initial_log_prob=None,
                 random=None, plot=False, var_names=None, update_interval=100):
                
        chain_q = np.empty((n_mcmc, len(initial_q)), dtype=float)
        log_prob_chain = np.empty(n_mcmc, dtype=float)

        if plot:
            chain_p = np.empty((n_mcmc, len(initial_q)), dtype=float)
            fig, ax = self._init_subplots(len(initial_q), 2, var_names)

            for n, (q, lp, p) in enumerate(self.sample(
            initial_q, n_mcmc, tune=tune, initial_log_prob=initial_log_prob,
            random=random)):
                chain_q[n] = q
                chain_p[n] = p
                log_prob_chain[n] = lp
                if n % update_interval == 0:
                    for j, axis in enumerate(ax):
                        # use axis.set_ydata later
                        axis[0].clear()
                        axis[1].clear()
                        axis[1].plot(chain_q[:n,j], chain_p[:n,j])
                        axis[1].plot(chain_q[n-1,j], chain_p[n-1,j], 'o')
                        axis[0].plot(chain_q[:n,j])
                    plt.pause(0.02)
                    plt.draw()
            return chain_q, log_prob_chain
        else:
            for n, (q, lp, p) in enumerate(self.sample(
                initial_q, n_mcmc, tune=tune, initial_log_prob=initial_log_prob,
                random=random)):
                    chain_q[n] = q
                    log_prob_chain[n] = lp
            return chain_q, log_prob_chain

    def _init_subplots(self, rows, columns, title=None):
        fig, ax = plt.subplots(rows, columns, squeeze=False)
        if title is not None:
            for axis, name in zip(ax, title):
                            axis[0].set_title(name)
        return fig, ax
