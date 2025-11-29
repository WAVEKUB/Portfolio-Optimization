# src/optimizers/de.py
import numpy as np
from ..utils import project_to_simplex_with_turnover, objective_neg_sharpe

def de_optimize(mu, Sigma, rf=0.0, pop=60, iters=600, F=0.7, CR=0.9, rs=None, current=None, max_turnover=1.0):
    rs = np.random.RandomState() if rs is None else rs
    n = len(mu)
    cur = current
    P = np.array([project_to_simplex_with_turnover(rs.rand(n), current=cur, max_turnover=max_turnover) for _ in range(pop)])
    fit = np.array([objective_neg_sharpe(w, mu, Sigma, rf) for w in P])
    hist = []
    for t in range(iters):
        for i in range(pop):
            idxs = [idx for idx in range(pop) if idx != i]
            a, b, c = P[rs.choice(idxs, 3, replace=False)]
            jrand = rs.randint(n)
            mutant = a + F * (b - c)
            trial = np.array([mutant[j] if (rs.rand() < CR or j == jrand) else P[i, j] for j in range(n)])
            trial = project_to_simplex_with_turnover(trial, current=cur, max_turnover=max_turnover)
            trial_fit = objective_neg_sharpe(trial, mu, Sigma, rf)
            if trial_fit < fit[i]:
                P[i] = trial; fit[i] = trial_fit
        if (t+1) % 25 == 0: hist.append(fit.min())
    best_idx = np.argmin(fit)
    return P[best_idx].copy(), fit[best_idx], np.array(hist)
