import numpy as np
import pandas as pd
import numpy.random as npr
import torch
from scipy.special import xlogy


def rbf_kernel(
    x1, x2, lengthscale_unconstrained, output_variance_unconstrained, diag=False
):

    lengthscale = torch.exp(lengthscale_unconstrained)
    output_variance = torch.exp(output_variance_unconstrained)

    if diag:
        diffs = x1 - x2
    else:
        diffs = x1.unsqueeze(-2) - x2.unsqueeze(-3)

    K = output_variance * torch.exp(
        -0.5 * torch.sum(torch.square(diffs / lengthscale), dim=-1)
    )
    return K


def rbf_kernel_numpy(x, xp, kernel_params):
    output_scale = np.exp(kernel_params[0])
    lengthscales = np.exp(kernel_params[1:])
    diffs = np.expand_dims(x / lengthscales, 1) - np.expand_dims(xp / lengthscales, 0)
    return output_scale * np.exp(-0.5 * np.sum(diffs**2, axis=2))


def matern12_kernel(
    x1, x2, lengthscale_unconstrained, output_variance_unconstrained, diag=False
):

    lengthscale = torch.exp(lengthscale_unconstrained)
    output_variance = torch.exp(output_variance_unconstrained)

    if diag:
        diffs = x1 - x2
    else:
        diffs = x1.unsqueeze(-2) - x2.unsqueeze(-3)
    eps = 1e-10
    dists = torch.sqrt(torch.sum(torch.square(diffs), dim=-1) + eps)

    return output_variance * torch.exp(-0.5 * dists / lengthscale)


def matern32_kernel(
    x1, x2, lengthscale_unconstrained, output_variance_unconstrained, diag=False
):

    lengthscale = torch.exp(lengthscale_unconstrained)
    output_variance = torch.exp(output_variance_unconstrained)

    if diag:
        diffs = x1 - x2
    else:
        diffs = x1.unsqueeze(-2) - x2.unsqueeze(-3)
    eps = 1e-10
    dists = torch.sqrt(torch.sum(torch.square(diffs), dim=-1) + eps)

    inner_term = np.sqrt(3.0) * dists / lengthscale
    K = output_variance * (1 + inner_term) * torch.exp(-inner_term)
    return K


def polar_warp(X, r, theta):
    return np.array([X[:, 0] + r * np.cos(theta), X[:, 1] + r * np.sin(theta)]).T


def get_st_coordinates(df):
    """
    Extracts spatial coordinates from ST data with index in 'AxB' type format.

    Return: pandas dataframe of coordinates
    """
    coor = []
    for spot in df.index:
        coordinates = spot.split("x")
        coordinates = [float(i) for i in coordinates]
        coor.append(coordinates)
    return np.array(coor)


def compute_distance(X1, X2):
    return np.mean(np.sqrt(np.sum((X1 - X2) ** 2, axis=1)))


def make_pinwheel(
    radial_std, tangential_std, num_classes, num_per_class, rate, rs=npr.RandomState(0)
):
    """Based on code by Ryan P. Adams."""
    rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

    features = rs.randn(num_classes * num_per_class, 2) * np.array(
        [radial_std, tangential_std]
    )
    features[:, 0] += 1
    labels = np.repeat(np.arange(num_classes), num_per_class)

    angles = rads[labels] + rate * np.exp(features[:, 0])
    rotations = np.stack(
        [np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)]
    )
    rotations = np.reshape(rotations.T, (-1, 2, 2))

    return np.einsum("ti,tij->tj", features, rotations)


class ConvergenceChecker(object):
    def __init__(self, span, dtp="float64"):
        self.span = span
        x = np.arange(span, dtype=dtp)
        x -= x.mean()
        X = np.column_stack((np.ones(shape=x.shape), x, x**2, x**3))
        self.U = np.linalg.svd(X, full_matrices=False)[0]

    def smooth(self, y):
        return self.U @ (self.U.T @ y)

    def subset(self, y, idx=-1):
        span = self.U.shape[0]
        lo = idx - span + 1
        if idx == -1:
            return y[lo:]
        else:
            return y[lo : (idx + 1)]

    def relative_change(self, y, idx=-1, smooth=True):
        y = self.subset(y, idx=idx)
        if smooth:
            y = self.smooth(y)
        prev = y[-2]
        return (y[-1] - prev) / (0.1 + abs(prev))

    def converged(self, y, tol=1e-4, **kwargs):
        return abs(self.relative_change(y, **kwargs)) < tol

    def relative_change_all(self, y, smooth=True):
        n = len(y)
        span = self.U.shape[0]
        cc = np.tile([np.nan], n)
        for i in range(span, n):
            cc[i] = self.relative_change(y, idx=i, smooth=smooth)
        return cc

    def converged_all(self, y, tol=1e-4, smooth=True):
        cc = self.relative_change_all(y, smooth=smooth)
        return np.abs(cc) < tol


# Function for computing size factors
def compute_size_factors(m):
    # given matrix m with samples in the columns
    # compute size factors

    sz = np.sum(m.values, axis=0)  # column sums (sum of counts in each cell)
    lsz = np.log(sz)

    # make geometric mean of sz be 1 for poisson
    sz_poisson = np.exp(lsz - np.mean(lsz))
    return sz_poisson


def poisson_deviance(X, sz):

    LP = X.values / sz  # recycling
    # import ipdb; ipdb.set_trace()
    LP[LP > 0] = np.log(LP[LP > 0])  # log transform nonzero elements only

    # Transpose to make features in cols, observations in rows
    X = X.T
    ll_sat = np.sum(np.multiply(X, LP.T), axis=0)
    feature_sums = np.sum(X, axis=0)
    ll_null = feature_sums * np.log(feature_sums / np.sum(sz))
    return 2 * (ll_sat - ll_null)


def deviance_feature_selection(X):

    # Remove cells without any counts
    X = X[np.sum(X, axis=1) > 0]

    # Compute size factors
    sz = compute_size_factors(X)

    # Compute deviances
    devs = poisson_deviance(X, sz)

    # Get associated gene names
    gene_names = X.index.values

    assert gene_names.shape[0] == devs.values.shape[0]

    return devs.values, gene_names


def deviance_residuals(x, theta, mu=None):
    """Computes deviance residuals for NB model with a fixed theta"""

    if mu is None:
        counts_sum0 = np.sum(x, axis=0, keepdims=True)
        counts_sum1 = np.sum(x, axis=1, keepdims=True)
        counts_sum = np.sum(x)
        # get residuals
        mu = counts_sum1 @ counts_sum0 / counts_sum

    def remove_negatives(sqrt_term):
        negatives_idx = sqrt_term < 0
        if np.any(negatives_idx):
            n_negatives = np.sum(negatives_idx)
            print(
                "Setting %u negative sqrt term values to 0 (%f%%)"
                % (n_negatives, n_negatives / np.product(sqrt_term.shape))
            )
            sqrt_term[negatives_idx] = 0

    if np.isinf(theta):  ### POISSON
        x_minus_mu = x - mu
        sqrt_term = 2 * (
            xlogy(x, x / mu) - x_minus_mu
        )  # xlogy(x,x/mu) computes xlog(x/mu) and returns 0 if x=0
        remove_negatives(sqrt_term)
        dev = np.sign(x_minus_mu) * np.sqrt(sqrt_term)
    else:  ### NEG BIN
        x_plus_theta = x + theta
        sqrt_term = 2 * (
            xlogy(x, x / mu) - (x_plus_theta) * np.log(x_plus_theta / (mu + theta))
        )  # xlogy(x,x/mu) computes xlog(x/mu) and returns 0 if x=0
        remove_negatives(sqrt_term)
        dev = np.sign(x - mu) * np.sqrt(sqrt_term)

    return dev


def pearson_residuals(counts, theta, clipping=True):
    """Computes analytical residuals for NB model with a fixed theta, clipping outlier residuals to sqrt(N)"""
    counts_sum0 = np.sum(counts, axis=0, keepdims=True)
    counts_sum1 = np.sum(counts, axis=1, keepdims=True)
    counts_sum = np.sum(counts)

    # get residuals
    mu = counts_sum1 @ counts_sum0 / counts_sum
    z = (counts - mu) / np.sqrt(mu + mu**2 / theta)

    # clip to sqrt(n)
    if clipping:
        n = counts.shape[0]
        z[z > np.sqrt(n)] = np.sqrt(n)
        z[z < -np.sqrt(n)] = -np.sqrt(n)

    return z


class LossNotDecreasingChecker:
    def __init__(self, max_epochs, atol=1e-2, window_size=10):
        self.max_epochs = max_epochs
        self.atol = atol
        self.window_size = window_size
        self.decrease_in_loss = np.zeros(max_epochs)
        self.average_decrease_in_loss = np.zeros(max_epochs)

    def check_loss(self, iternum, loss_trace):

        if iternum >= 1:
            self.decrease_in_loss[iternum] = (
                loss_trace[iternum - 1] - loss_trace[iternum]
            )
            if iternum >= self.window_size:
                self.average_decrease_in_loss[iternum] = np.mean(
                    self.decrease_in_loss[iternum - self.window_size + 1 : iternum]
                )
                has_converged = self.average_decrease_in_loss[iternum] < self.atol
                return has_converged

        return False
