# src/optimizers/sa.py
import numpy as np
from ..utils import project_to_simplex_with_turnover, objective_neg_sharpe

def sa_optimize(mu, Sigma, rf=0.0, iters=4000, T0=0.2, Tend=1e-3, step=0.05, rs=None, current=None, max_turnover=1.0):
    rs = np.random.RandomState() if rs is None else rs
    n = len(mu)
    cur = current
    x = project_to_simplex_with_turnover(rs.rand(n), current=cur, max_turnover=max_turnover)
    fx = objective_neg_sharpe(x, mu, Sigma, rf)
    best, best_f = x.copy(), fx
    hist = []
    for t in range(1, iters+1):
        T = T0 * (Tend / T0) ** (t / iters)
        cand = project_to_simplex_with_turnover(x + rs.normal(scale=step, size=n), current=cur, max_turnover=max_turnover)
        fc = objective_neg_sharpe(cand, mu, Sigma, rf)
        if (fc < fx) or (rs.rand() < np.exp(-(fc - fx) / (T + 1e-12))):
            x, fx = cand, fc
            if fx < best_f:
                best, best_f = x.copy(), fx
        if t % 100 == 0: hist.append(best_f)
    return best, best_f, np.array(hist)
