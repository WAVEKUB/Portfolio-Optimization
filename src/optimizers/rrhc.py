# src/optimizers/rrhc.py
import numpy as np
from ..utils import project_to_simplex_with_turnover, objective_neg_sharpe

def rrhc_optimize(mu, Sigma, rf=0.0, restarts=20, iters=300, neigh=40, step=0.05, rs=None, current=None, max_turnover=1.0):
    rs = np.random.RandomState() if rs is None else rs
    n = len(mu)
    cur = current
    best, best_f = None, np.inf
    hist = []
    for r in range(restarts):
        x = project_to_simplex_with_turnover(rs.rand(n), current=cur, max_turnover=max_turnover)
        fx = objective_neg_sharpe(x, mu, Sigma, rf)
        for t in range(iters):
            cand = x + rs.normal(scale=step, size=(neigh, n))
            cand = np.array([project_to_simplex_with_turnover(c, current=cur, max_turnover=max_turnover) for c in cand])
            fvals = np.array([objective_neg_sharpe(c, mu, Sigma, rf) for c in cand])
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < fx:
                x, fx = cand[min_idx], fvals[min_idx]
            else:
                break
        if fx < best_f:
            best, best_f = x.copy(), fx
        hist.append(best_f)
    return best, best_f, np.array(hist)
