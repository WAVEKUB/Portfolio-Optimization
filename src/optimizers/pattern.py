# src/optimizers/pattern.py
import numpy as np
from ..utils import project_to_simplex_with_turnover, objective_neg_sharpe

def pattern_search_optimize(mu, Sigma, rf=0.0, iters=200, init_step=0.2, step_shrink=0.5, tol=1e-4, rs=None, current=None, max_turnover=1.0):
    rs = np.random.RandomState() if rs is None else rs
    n = len(mu)
    cur = current
    x = project_to_simplex_with_turnover(rs.rand(n), current=cur, max_turnover=max_turnover)
    fx = objective_neg_sharpe(x, mu, Sigma, rf)
    step = init_step
    hist = []
    for t in range(iters):
        improved = False
        for j in range(n):
            for direction in [+1, -1]:
                cand = x.copy()
                cand[j] += direction * step
                cand = project_to_simplex_with_turnover(cand, current=cur, max_turnover=max_turnover)
                fc = objective_neg_sharpe(cand, mu, Sigma, rf)
                if fc < fx - 1e-12:
                    x, fx = cand, fc
                    improved = True
        if not improved:
            step *= step_shrink
            if step < tol:
                break
        if (t+1) % 5 == 0: hist.append(fx)
    return x, fx, np.array(hist)
