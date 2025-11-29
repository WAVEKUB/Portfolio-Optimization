# src/optimizers/harmony.py
import numpy as np
from ..utils import project_to_simplex_with_turnover, objective_neg_sharpe

def harmony_search(mu, Sigma, rf=0.0, HMS=50, iters=600, HMCR=0.9, PAR=0.3, bw=0.05, rs=None, current=None, max_turnover=1.0):
    rs = np.random.RandomState() if rs is None else rs
    n = len(mu)
    cur = current
    HM = np.array([project_to_simplex_with_turnover(rs.rand(n), current=cur, max_turnover=max_turnover) for _ in range(HMS)])
    HM_fit = np.array([objective_neg_sharpe(w, mu, Sigma, rf) for w in HM])
    hist = []
    for t in range(iters):
        new = np.zeros(n)
        for j in range(n):
            if rs.rand() < HMCR:
                idx = rs.randint(HMS); val = HM[idx, j]
                if rs.rand() < PAR: val += rs.uniform(-bw, bw)
                new[j] = val
            else:
                new[j] = rs.rand()
        new = project_to_simplex_with_turnover(new, current=cur, max_turnover=max_turnover)
        new_fit = objective_neg_sharpe(new, mu, Sigma, rf)
        worst = np.argmax(HM_fit)
        if new_fit < HM_fit[worst]:
            HM[worst] = new; HM_fit[worst] = new_fit
        if (t+1) % 25 == 0: hist.append(HM_fit.min())
    best_idx = np.argmin(HM_fit)
    return HM[best_idx].copy(), HM_fit[best_idx], np.array(hist)
