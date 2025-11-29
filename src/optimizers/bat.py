# src/optimizers/bat.py
import numpy as np
from ..utils import project_to_simplex_with_turnover, objective_neg_sharpe

def bat_optimize(mu, Sigma, rf=0.0, n_bats=50, iters=600, fmin=0.0, fmax=2.0,
                 alpha=0.9, gamma=0.9, rs=None, current=None, max_turnover=1.0):
    rs = np.random.RandomState() if rs is None else rs
    n = len(mu)
    cur = current
    Q = np.array([project_to_simplex_with_turnover(rs.rand(n), current=cur, max_turnover=max_turnover) for _ in range(n_bats)])
    v = np.zeros_like(Q)
    f = np.zeros(n_bats)
    A = np.ones(n_bats)
    r = rs.rand(n_bats)
    fitness = np.array([objective_neg_sharpe(q, mu, Sigma, rf) for q in Q])
    best_idx = np.argmin(fitness)
    best = Q[best_idx].copy()
    best_fit = fitness[best_idx]
    hist = []
    for t in range(iters):
        for i in range(n_bats):
            f[i] = fmin + (fmax - fmin) * rs.rand()
            v[i] = v[i] + (Q[i] - best) * f[i]
            cand = project_to_simplex_with_turnover(Q[i] + v[i], current=cur, max_turnover=max_turnover)
            if rs.rand() > r[i]:
                cand = project_to_simplex_with_turnover(best + rs.normal(scale=0.01, size=n) * np.mean(A), current=cur, max_turnover=max_turnover)
            cand_fit = objective_neg_sharpe(cand, mu, Sigma, rf)
            if (cand_fit <= fitness[i]) and (rs.rand() < A[i]):
                Q[i] = cand
                fitness[i] = cand_fit
                A[i] *= alpha
                r[i] = r[i] * (1 - np.exp(-gamma * t / (iters + 1.0)))
            if fitness[i] < best_fit:
                best_fit = fitness[i]; best = Q[i].copy()
        if (t+1) % 25 == 0: hist.append(best_fit)
    return best, best_fit, np.array(hist)
