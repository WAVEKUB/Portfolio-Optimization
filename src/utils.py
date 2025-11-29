# src/utils.py
import numpy as np

def project_to_simplex(w):
    if np.all(w >= 0) and abs(w.sum() - 1) < 1e-10:
        return w
    u = np.sort(w)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(w)+1) > (cssv - 1))[0].max()
    theta = (cssv[rho] - 1.0) / (rho + 1)
    w_proj = np.maximum(w - theta, 0.0)
    s = w_proj.sum()
    if s == 0:
        return np.ones_like(w_proj) / len(w_proj)
    return w_proj / s

def project_to_simplex_with_turnover(w, current=None, max_turnover=1.0):
    """Project w to simplex, but limit L1 turnover vs current (if provided).
       If max_turnover==0 and current provided -> returns current (no trading).
    """
    # basic projection
    w_proj = project_to_simplex(w)
    if current is None or max_turnover is None or max_turnover >= 2.0:
        return w_proj
    current = np.asarray(current, dtype=float)
    # if no trading allowed
    if max_turnover <= 0.0:
        return current.copy()
    delta = w_proj - current
    turnover = np.abs(delta).sum()
    if turnover <= max_turnover:
        return w_proj
    # scale changes to fit allowed turnover, then re-project to ensure feasibility
    scale = max_turnover / (turnover + 1e-12)
    w_limited = project_to_simplex(current + delta * scale)
    return w_limited

def portfolio_stats(w, mu, Sigma, rf=0.0):
    ret = float(np.dot(w, mu))
    vol = float(np.sqrt(np.dot(w, np.dot(Sigma, w))))
    sharpe = (ret - rf) / (vol + 1e-12)
    return ret, vol, sharpe

def objective_neg_sharpe(w, mu, Sigma, rf=0.0):
    _, _, s = portfolio_stats(w, mu, Sigma, rf)
    return -s
