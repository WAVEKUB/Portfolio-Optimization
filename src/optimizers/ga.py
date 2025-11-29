# src/optimizers/ga.py
import numpy as np
from ..utils import project_to_simplex_with_turnover, objective_neg_sharpe

def ga_optimize(mu, Sigma, rf=0.0, pop=80, iters=500, elite_frac=0.2, cx_prob=0.8, mut_prob=0.3, mut_scale=0.1, rs=None, current=None, max_turnover=1.0):
    rs = np.random.RandomState() if rs is None else rs
    n = len(mu)
    cur = current
    P = np.array([project_to_simplex_with_turnover(rs.rand(n), current=cur, max_turnover=max_turnover) for _ in range(pop)])
    fit = np.array([objective_neg_sharpe(w, mu, Sigma, rf) for w in P])
    hist = []
    elite_k = max(2, int(elite_frac * pop))
    for t in range(iters):
        idx = np.argsort(fit)
        P = P[idx]; fit = fit[idx]
        elites = P[:elite_k].copy()
        def select():
            i, j = rs.randint(pop, size=2)
            return P[i] if fit[i] < fit[j] else P[j]
        children = []
        while len(children) < pop - elite_k:
            p1, p2 = select(), select()
            if rs.rand() < cx_prob:
                alpha = rs.rand()
                child = alpha * p1 + (1 - alpha) * p2
            else:
                child = p1.copy()
            if rs.rand() < mut_prob:
                child += rs.normal(scale=mut_scale, size=n)
            children.append(project_to_simplex_with_turnover(child, current=cur, max_turnover=max_turnover))
        P = np.vstack([elites, np.array(children)])
        fit = np.array([objective_neg_sharpe(w, mu, Sigma, rf) for w in P])
        if (t+1) % 25 == 0: hist.append(fit.min())
    best_idx = np.argmin(fit)
    return P[best_idx].copy(), fit[best_idx], np.array(hist)
