# src/optimizers/acor.py
import numpy as np
from ..utils import project_to_simplex_with_turnover, objective_neg_sharpe

def acor_optimize(mu, Sigma, rf=0.0, archive_size=50, n_ants=40, iters=600, q=0.1, xi=0.85, rs=None, current=None, max_turnover=1.0):
    rs = np.random.RandomState() if rs is None else rs
    n = len(mu)
    cur = current
    archive = np.array([project_to_simplex_with_turnover(rs.rand(n), current=cur, max_turnover=max_turnover) for _ in range(archive_size)])
    fvals = np.array([objective_neg_sharpe(w, mu, Sigma, rf) for w in archive])
    idx = np.argsort(fvals)
    archive, fvals = archive[idx], fvals[idx]
    ranks = np.arange(archive_size)
    weights = np.exp(- (ranks**2) / (2 * (q * archive_size)**2))
    weights = weights / weights.sum()
    def sigma_vec():
        s = np.zeros(n)
        for j in range(n):
            val = 0.0
            for r in range(archive_size):
                val += abs(archive[r, j] - archive[0, j])
            s[j] = xi * val / (archive_size - 1 + 1e-12)
        s = np.maximum(s, 1e-4)
        return s
    best = archive[0].copy(); best_f = fvals[0]
    history = []
    for t in range(iters):
        sig = sigma_vec()
        new_solutions = []
        for _ in range(n_ants):
            r_idx = rs.choice(archive_size, p=weights)
            mu_vec = archive[r_idx]
            cand = mu_vec + rs.normal(scale=sig, size=n)
            cand = project_to_simplex_with_turnover(cand, current=cur, max_turnover=max_turnover)
            new_solutions.append(cand)
        new_solutions = np.array(new_solutions)
        new_fvals = np.array([objective_neg_sharpe(w, mu, Sigma, rf) for w in new_solutions])
        archive = np.vstack([archive, new_solutions])
        fvals = np.concatenate([fvals, new_fvals])
        idx = np.argsort(fvals)
        archive, fvals = archive[idx][:archive_size], fvals[idx][:archive_size]
        if fvals[0] < best_f:
            best_f = fvals[0]; best = archive[0].copy()
        if (t+1) % 25 == 0: history.append(best_f)
    return best, best_f, np.array(history)
