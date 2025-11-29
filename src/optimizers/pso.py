# src/optimizers/pso.py
import numpy as np
from ..utils import project_to_simplex_with_turnover, objective_neg_sharpe

def pso_optimize(mu, Sigma, rf=0.0, swarm=60, iters=600, w_inertia=0.7, c1=1.6, c2=1.6, rs=None, current=None, max_turnover=1.0):
    rs = np.random.RandomState() if rs is None else rs
    n = len(mu)
    cur = current
    X = np.array([project_to_simplex_with_turnover(rs.rand(n), current=cur, max_turnover=max_turnover) for _ in range(swarm)])
    V = rs.normal(scale=0.05, size=(swarm, n))
    fit = np.array([objective_neg_sharpe(x, mu, Sigma, rf) for x in X])
    pbest = X.copy(); pbest_fit = fit.copy()
    g_idx = np.argmin(fit); gbest = X[g_idx].copy(); gbest_fit = fit[g_idx]
    hist = []
    for t in range(iters):
        r1, r2 = rs.rand(swarm, n), rs.rand(swarm, n)
        V = w_inertia * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        X = np.array([project_to_simplex_with_turnover(x + v, current=cur, max_turnover=max_turnover) for x, v in zip(X, V)])
        fit = np.array([objective_neg_sharpe(x, mu, Sigma, rf) for x in X])
        improved = fit < pbest_fit
        pbest[improved] = X[improved]
        pbest_fit[improved] = fit[improved]
        g_idx = np.argmin(fit)
        if fit[g_idx] < gbest_fit:
            gbest = X[g_idx].copy(); gbest_fit = fit[g_idx]
        if (t+1) % 25 == 0: hist.append(gbest_fit)
    return gbest, gbest_fit, np.array(hist)
