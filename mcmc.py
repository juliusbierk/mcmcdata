import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from multiprocessing import Pool
from functools import partial

s = 0.1


@njit
def v(x, a, b):
    return - a * np.sign(x) * np.abs(x) ** b


@njit
def simulation(a=0.55, b=1.5, n=100):
    x0 = 0
    xs = np.empty(n)
    for i in range(n):
        x0 += v(x0, a, b) + s * np.random.randn()
        xs[i] = x0
    return xs


@njit
def log_likelihood(x, a, b):
    x_old = x[:-1]
    x_new = x[1:]

    x_old_v = x_old + v(x_old, a, b)

    log_p = np.sum(-(x_new - x_old_v) ** 2 / (2 * s ** 2))
    return log_p


def maximum_likelihood(x):
    from scipy.optimize import minimize
    res = minimize(lambda p: -log_likelihood(x, p[0], p[1]),
                   x0=(1.0, 1.0), method="Nelder-Mead")
    print(res.x)


@njit
def mcmc(x, n=100_000):
    a0 = 1.0
    b0 = 1.0
    pars = np.empty((n, 2))
    ll = log_likelihood(x, a0, b0)
    accepted = 0
    for i in range(n):
        an = a0 + 0.05 * np.random.randn()
        bn = b0 + 0.05 * np.random.randn()
        new_ll = log_likelihood(x, an, bn)
        alpha = np.exp(new_ll - ll)
        if alpha > 1 or np.random.random() < alpha:
            accepted += 1
            a0 = an
            b0 = bn
            ll = new_ll
        pars[i, 0] = a0
        pars[i, 1] = b0
    print('Done, acceptance probability', accepted / n)
    return pars[n // 2:]


def mcmc_chains(x, n=100_000, chains=4):
    pars = np.empty((chains, n // 2, 2))
    with Pool(processes=4) as pool:
        runs = pool.imap(partial(mcmc, n=n), [x] * chains)

        for c, res in enumerate(runs):
            print(f'Finished chain {c + 1}')
            pars[c, :, :] = res
    return pars


def main_2():
    x = simulation(n=5000)
    pars = mcmc_chains(x)
    plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1)
    for c in range(pars.shape[0]):
        plt.hist(pars[c, :, 0], alpha=0.25)
    plt.xlabel('a')

    plt.subplot(1, 3, 2)
    for c in range(pars.shape[0]):
        plt.hist(pars[c, :, 1], alpha=0.25)
    plt.xlabel('b')

    plt.subplot(1, 3, 3)
    for c in range(pars.shape[0]):
        plt.plot(pars[c, :, 0], pars[c, :, 1], '.', alpha=0.2, markersize=0.25)
    plt.xlabel('a')
    plt.ylabel('b')

    plt.tight_layout()
    plt.show()


def main_1():
    x = simulation(n=5000)
    pars = mcmc(x)
    plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1)
    plt.title(f'a = {np.mean(pars[:, 0]):0.2f} +- {np.std(pars[:, 0]):0.2f}')
    plt.hist(pars[:, 0], bins=100)
    plt.subplot(1, 3, 2)
    plt.title(f'b = {np.mean(pars[:, 1]):0.2f} +- {np.std(pars[:, 1]):0.2f}')
    plt.hist(pars[:, 1], bins=100)
    plt.subplot(1, 3, 3)
    plt.plot(pars[:, 0], pars[:, 1], '.', alpha=0.5, markersize=0.2)
    plt.xlabel('a')
    plt.ylabel('b')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main_2()
