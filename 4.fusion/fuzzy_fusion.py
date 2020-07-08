
import os
import sys
import glob
import numpy as np
import time
import tqdm

from sklearn.metrics import accuracy_score
from sympy import solve, Symbol


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=1)[:, None]


def norm(x, Min=None, Max=None):
    if Min is None:
        Min = x.min()
    if Max is None:
        Max = x.max()
    return (x - Min) / (Max - Min)


def min_max(x):
    return (x - np.min(x, axis=1)[:, None]) / (np.max(x, axis=1) - np.min(x, axis=1))[:, None]


def get_lambda(fm):
    """Computes the lambda necessary for lambda fuzzy measures given the singletons."""

    x = Symbol('x', real=True)
    eqn = (1 + x*fm[0])
    for i in range(1, fm.shape[0]):
        eqn = eqn * (1 + x*fm[i])
    eqn = eqn - x - 1
    # print(eqn)

    # eqn = (1 + x*fm[0])*(1 + x*fm[1])*(1 + x*fm[2]) - x - 1
    sol = solve(eqn, x)
    # print(sol)
    sol_lambda = sol[-1]
    return sol_lambda

def fuzzyFusion_3(mix, fm, lambda_value):
    """Computes the fusion of 3 different sources given an array of diferent predictions
        mix = [source, samples, confidences for each class],
        an array of sigletion fuzzy measures fm (weights) and the lambda"""
    idx = np.argsort(mix, axis=0)[::-1, ...]
    h = np.sort(mix, axis=0)[::-1, ...]

    FM = np.ones_like(h)
    for i in range(mix.shape[0]):
        FM[i, ...] *= fm[i]

    # sort fuzzy measures the same way as mix
    ind = np.unravel_index(idx, idx.shape, order='F')
    FM = FM[ind]

    ff = h[0, ...]*FM[0, ...]
    # A = np.minimum(1, FM[1, :] + FM[0, :])
    A = FM[1, ...] + FM[0, ...] + lambda_value*FM[1, ...]*FM[0, ...]
    ff = ff + h[1, ...]*(A - FM[0, ...])
    ff = ff + h[2, ...]*(1 - A)

    return ff


def fuzzyFusion_2(mix, fm, lambda_value):
    """Computes the fusion of 2 different sources given an array of diferent predictions
        mix = [source, samples, confidences for each class],
        an array of sigletion fuzzy measures fm (weights) and the lambda"""

    idx = np.argsort(mix, axis=0)[::-1, ...]
    h = np.sort(mix, axis=0)[::-1, ...]

    FM = np.ones_like(h)
    for i in range(mix.shape[0]):
        FM[i, ...] *= fm[i]

    # sort fuzzy measures the same way as mix
    ind = np.unravel_index(idx, idx.shape, order='F')
    FM = FM[ind]

    ff = h[0, ...]*FM[0, ...]
    ff = ff + h[1, ...]*(1 - FM[0, ...])

    return ff


def get_acc_ff(w, pred, y):
    w = np.array(w)
    fm = w/10  # (sum(w)*2)

    lambda_v = None
    if fm.shape[0] > 2:
        lambda_v = float(get_lambda(fm))

    # print(w, lambda_v)
    streams = np.array(pred)
    ff = np.zeros_like(pred[0])

    ff = fuzzyFusion_3(streams, fm, lambda_v)

    y_pred = np.argmax(ff, -1)
    acc = accuracy_score(y, y_pred)
    return acc


def get_acc(w, pred, y):
    avg_predictions = np.zeros_like(pred[0])

    for i in range(len(w)):
        if sum(w) > 0:
            w_i = w[i] / sum(w)
        else:
            w_i = w[i]
        avg_predictions += pred[i]*w_i

    y_pred = np.argmax(avg_predictions, -1)
    acc = accuracy_score(y, y_pred)
    return acc


def fuzzy_fusion(data, w):
    data = np.array(data)
    w = np.array(w)
    fm = w / (sum(w)*2)

    n = w.shape[0]
    lambda_v = float(get_lambda(fm))

    idx = np.argsort(data, axis=0)[::-1, ...]
    h = np.sort(data, axis=0)[::-1, ...]

    FM = np.ones_like(h)
    for i in range(data.shape[0]):
        FM[i, ...] *= fm[i]

    # sort fuzzy measures the same way as mix
    ind = np.unravel_index(idx, idx.shape, order='F')
    FM = FM[ind]

    ff = h[0, ...] * FM[0, ...]
    A0 = FM[0, ...]

    for i in range(1, n-1):
        A = A0 + FM[i, ...] + lambda_v*A0*FM[i, ...]
        ff = ff + h[i, ...] * (A - A0)
        A0 = A
    ff = ff + h[n-1, ...] * (1 - A0)
    return ff
