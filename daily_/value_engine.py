# >>>M37_FILE: value_engine.py
# value_engine.py
# 37号 · Value Engine —— 多模型融合 + 市场先验 + 四分盘 + 真实EV/Kelly + 质量工具

import math
import numpy as np
from typing import Dict, Tuple

OU_OR_MIN, OU_OR_MAX = 1.02, 1.18
X1X2_OR_MIN, X1X2_OR_MAX = 1.02, 1.20

def set_seed(seed: int | None):
    if seed is None:
        return
    try:
        np.random.seed(int(seed))
    except Exception:
        pass

def implied_probs_from_odds(odds: Dict[str, float]) -> Dict[str, float]:
    inv = {k: (1.0/float(v) if v and float(v)>1 else 0.0) for k,v in odds.items()}
    s = sum(inv.values()) or 1.0
    return {k: inv[k]/s for k in inv}

def overround_from_odds(odds: Dict[str, float]) -> float | None:
    try:
        inv = [1.0/float(v) for v in odds.values() if v and float(v)>1]
        return float(sum(inv)) if inv else None
    except Exception:
        return None

def poisson_1x2_closed(lam_home: float, lam_away: float, max_goals: int = 10) -> Dict[str,float]:
    ph = [0.0]*(max_goals+1); pa = [0.0]*(max_goals+1)
    ph[0] = math.exp(-lam_home); pa[0] = math.exp(-lam_away)
    for i in range(1, max_goals+1):
        ph[i] = ph[i-1]*lam_home/i
        pa[i] = pa[i-1]*lam_away/i
    p_home = p_draw = p_away = 0.0
    for h in range(max_goals+1):
        for a in range(max_goals+1):
            p = ph[h]*pa[a]
            if h>a: p_home += p
            elif h==a: p_draw += p
            else: p_away += p
    s = p_home+p_draw+p_away
    if s>0:
        p_home/=s; p_draw/=s; p_away/=s
    return {"p_home":p_home,"p_draw":p_draw,"p_away":p_away}

def ou_quarter_probs(line: float, lam_total: float):
    base = math.floor(float(line) + 1e-9)
    frac = round(float(line) - base, 2)
    def tail_ge(k, lam):
        if k<=0: return 1.0
        pmf = math.exp(-lam); s = pmf
        for n in range(1, k):
            pmf *= lam/n; s += pmf
        return max(0.0, 1.0 - s)
    def cum_le(k, lam):
        if k<0: return 0.0
        pmf = math.exp(-lam); s = pmf
        for n in range(1, k+1):
            pmf *= lam/n; s += pmf
        return min(1.0, s)
    if frac==0.0:
        return tail_ge(base+1, lam_total), cum_le(base-1, lam_total)
    if frac==0.5:
        return tail_ge(base+1, lam_total), cum_le(base,   lam_total)
    if frac==0.25:
        pO = 0.5*tail_ge(base+1, lam_total) + 0.5*tail_ge(base+1, lam_total)
        pU = 0.5*cum_le(base-1, lam_total)   + 0.5*cum_le(base,   lam_total)
        return pO, pU
    if frac==0.75:
        pO = 0.5*tail_ge(base+1, lam_total) + 0.5*tail_ge(base+2, lam_total)
        pU = 0.5*cum_le(base,   lam_total)   + 0.5*cum_le(base+1, lam_total)
        return pO, pU
    nearest_half = base + (0.5 if frac<0.5 else 1.5)
    return ou_quarter_probs(nearest_half, lam_total)

def blend_1x2(models: Dict[str, Dict[str,float]], weights: Dict[str,float]) -> Dict[str,float]:
    keys = ["p_home","p_draw","p_away"]
    acc = {k:0.0 for k in keys}
    wsum = 0.0
    for name, probs in models.items():
        w = float(weights.get(name, 0.0))
        if w<=0:
            continue
        for k in keys:
            acc[k] += w*float(probs.get(k, 0.0))
        wsum += w
    if wsum<=0:
        return next(iter(models.values()))
    out = {k: acc[k]/wsum for k in keys}
    s = sum(out.values()) or 1.0
    return {k: out[k]/s for k in keys}

def ev_kelly(p: float, odds: float, k_fraction: float=1.0):
    try:
        p = float(p); o = float(odds)
        if not (0<=p<=1) or o<=1: return None, None
        ev = p*o - 1
        num = (p*o - (1-p)); den = (o - 1.0)
        f = num/den if den>0 else None
        if f is None or f<=0: return round(ev,6), None
        return round(ev,6), round(min(1.0, max(0.0, f*k_fraction)), 6)
    except Exception:
        return None, None

# <<<M37_FILE: value_engine.py
