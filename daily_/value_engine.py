# >>>M37_FILE: value_engine.py
# value_engine.py
# 37号 · Value Engine —— 多模型融合 + 市场先验 + 四分盘 + 真实EV/Kelly + 质量工具

import math
import numpy as np
from typing import Dict, Tuple

OU_OR_MIN, OU_OR_MAX = 1.02, 1.18
X1X2_OR_MIN, X1X2_OR_MAX = 1.02, 1.20

LEAGUE_TIER_TOP = "top"
LEAGUE_TIER_OTHER = "other"

TOP_LEAGUE_IDS = {
    2,    # UEFA Champions League
    3,    # UEFA Europa League
    39,   # Premier League
    61,   # Ligue 1
    78,   # Bundesliga
    135,  # Serie A
    140,  # La Liga
    848,  # Europa Conference League
}

TOP_LEAGUE_KEYWORDS = {
    ("england", "premier league"),
    ("spain", "la liga"),
    ("spain", "primera division"),
    ("germany", "bundesliga"),
    ("italy", "serie a"),
    ("france", "ligue 1"),
}

EURO_CUP_KEYWORDS = {
    "champions league",
    "europa league",
    "europa conference",
    "uefa conference",
}

MARKET_POLICIES = {
    LEAGUE_TIER_TOP: {
        "1x2": {
            "keep_min": 0.02,
            "keep_max": 0.06,
            "drop": 0.12,
            "overround_range": (1.02, 1.12),
            "min_bookmakers": 6,
            "max_update_min": 10.0,
            "high_ev_review": 0.15,
            "high_ev_min_bookmakers": 8,
            "high_ev_max_update": 5.0,
            "high_ev_min_vi": 0.02,
        },
        "ou": {
            "keep_min": 0.02,
            "keep_max": 0.06,
            "drop": 0.12,
            "overround_range": (1.02, 1.12),
            "min_bookmakers": 6,
            "max_update_min": 10.0,
            "high_ev_review": 0.15,
            "high_ev_min_bookmakers": 8,
            "high_ev_max_update": 5.0,
            "high_ev_min_vi": 0.02,
        },
        "ah": {
            "keep_min": 0.02,
            "keep_max": 0.06,
            "drop": 0.12,
            "overround_range": (1.02, 1.12),
            "min_bookmakers": 6,
            "max_update_min": 10.0,
            "high_ev_review": 0.15,
            "high_ev_min_bookmakers": 8,
            "high_ev_max_update": 5.0,
            "high_ev_min_vi": 0.02,
        },
        "derivative": {
            "keep_min": 0.05,
            "keep_max": 0.12,
            "drop": 0.25,
            "overround_range": (1.02, 1.20),
            "min_bookmakers": 6,
            "max_update_min": 10.0,
            "high_ev_review": 0.18,
            "high_ev_min_bookmakers": 8,
            "high_ev_max_update": 5.0,
            "high_ev_min_vi": 0.02,
        },
    },
    LEAGUE_TIER_OTHER: {
        "1x2": {
            "keep_min": 0.04,
            "keep_max": 0.10,
            "drop": 0.18,
            "overround_range": (1.02, 1.12),
            "min_bookmakers": 8,
            "max_update_min": 10.0,
            "high_ev_review": 0.15,
            "high_ev_min_bookmakers": 8,
            "high_ev_max_update": 5.0,
            "high_ev_min_vi": 0.02,
        },
        "ou": {
            "keep_min": 0.04,
            "keep_max": 0.10,
            "drop": 0.18,
            "overround_range": (1.02, 1.12),
            "min_bookmakers": 8,
            "max_update_min": 10.0,
            "high_ev_review": 0.15,
            "high_ev_min_bookmakers": 8,
            "high_ev_max_update": 5.0,
            "high_ev_min_vi": 0.02,
        },
        "ah": {
            "keep_min": 0.04,
            "keep_max": 0.10,
            "drop": 0.18,
            "overround_range": (1.02, 1.12),
            "min_bookmakers": 8,
            "max_update_min": 10.0,
            "high_ev_review": 0.15,
            "high_ev_min_bookmakers": 8,
            "high_ev_max_update": 5.0,
            "high_ev_min_vi": 0.02,
        },
        "derivative": {
            "keep_min": 0.05,
            "keep_max": 0.12,
            "drop": 0.25,
            "overround_range": (1.02, 1.20),
            "min_bookmakers": 8,
            "max_update_min": 10.0,
            "high_ev_review": 0.18,
            "high_ev_min_bookmakers": 8,
            "high_ev_max_update": 5.0,
            "high_ev_min_vi": 0.02,
        },
    },
}


def _norm_text(text) -> str:
    return str(text or "").strip().lower()


def classify_league_tier(league_id: int | None, league_name: str | None, country: str | None) -> str:
    lid = None
    try:
        lid = int(league_id) if league_id is not None else None
    except (TypeError, ValueError):
        lid = None
    if lid in TOP_LEAGUE_IDS:
        return LEAGUE_TIER_TOP
    name = _norm_text(league_name)
    country_norm = _norm_text(country)
    for country_key, keyword in TOP_LEAGUE_KEYWORDS:
        if keyword in name and (not country_key or country_norm == country_key):
            return LEAGUE_TIER_TOP
    for keyword in EURO_CUP_KEYWORDS:
        if keyword in name:
            return LEAGUE_TIER_TOP
    return LEAGUE_TIER_OTHER


def kelly_cap_for_tier(tier: str) -> float:
    return 0.08 if tier == LEAGUE_TIER_TOP else 0.05


def market_policy(tier: str, market: str) -> Dict[str, float]:
    tier_policies = MARKET_POLICIES.get(tier, MARKET_POLICIES[LEAGUE_TIER_OTHER])
    return tier_policies.get(market, tier_policies.get("ou", {}))


def value_index(ev: float | None, kelly: float | None) -> float | None:
    if ev is None or kelly is None:
        return None
    try:
        ev_f = float(ev)
        k_f = float(kelly)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(ev_f) or not math.isfinite(k_f) or k_f < 0:
        return None
    return float(ev_f * math.sqrt(max(0.0, k_f)))


def evaluate_ev_market(
    ev: float | None,
    kelly: float | None,
    market: str,
    tier: str,
    *,
    min_bookmakers: int | None = None,
    overround: float | None = None,
    update_age: float | None = None,
) -> Dict[str, object]:
    result = {"ev": None, "kelly": None, "value_index": None, "tag": "invalid", "reasons": tuple()}
    if ev is None or kelly is None:
        return result
    try:
        ev_val = float(ev)
        kelly_val = float(kelly)
    except (TypeError, ValueError):
        return result
    if not math.isfinite(ev_val) or not math.isfinite(kelly_val) or kelly_val <= 0:
        return result

    cap = kelly_cap_for_tier(tier)
    kelly_val = min(cap, max(0.0, kelly_val))
    if kelly_val <= 0:
        return result

    policy = market_policy(tier, market)
    keep_min = float(policy.get("keep_min", 0.0))
    keep_max = float(policy.get("keep_max", keep_min))
    drop = float(policy.get("drop", keep_max))

    if ev_val > drop:
        reason = f"ev>{drop:.1%}"
        return {"ev": None, "kelly": None, "value_index": None, "tag": "reject", "reasons": (reason,)}

    if ev_val < keep_min:
        tag = "low"
    elif ev_val <= keep_max:
        tag = "keep"
    else:
        tag = "review"

    min_req = int(policy.get("min_bookmakers", 6 if tier == LEAGUE_TIER_TOP else 8))
    reasons = []
    if min_bookmakers is not None and min_bookmakers < min_req:
        reasons.append("bookmakers")

    try:
        overround_val = float(overround) if overround is not None else None
    except (TypeError, ValueError):
        overround_val = None
    lo, hi = policy.get("overround_range", (1.02, 1.12))
    if overround_val is not None and not (lo <= overround_val <= hi):
        reasons.append("overround")

    try:
        upd = float(update_age) if update_age is not None else None
    except (TypeError, ValueError):
        upd = None
    max_update = float(policy.get("max_update_min", 10.0))
    if upd is not None and upd > max_update:
        reasons.append("stale")

    if reasons:
        return {"ev": None, "kelly": None, "value_index": None, "tag": "reject", "reasons": tuple(reasons)}

    vi = value_index(ev_val, kelly_val)
    high_ev_cut = policy.get("high_ev_review")
    if high_ev_cut is not None and ev_val > float(high_ev_cut):
        guard_reasons = []
        guard_books = int(policy.get("high_ev_min_bookmakers", min_req))
        if min_bookmakers is None or min_bookmakers < guard_books:
            guard_reasons.append("hi_ev_books")
        guard_max_update = float(policy.get("high_ev_max_update", max_update))
        if upd is None or upd > guard_max_update:
            guard_reasons.append("hi_ev_stale")
        guard_vi = float(policy.get("high_ev_min_vi", 0.02))
        if vi is None or vi < guard_vi:
            guard_reasons.append("hi_ev_vi")
        if guard_reasons:
            return {"ev": None, "kelly": None, "value_index": None, "tag": "reject", "reasons": tuple(guard_reasons)}

    return {
        "ev": round(ev_val, 6),
        "kelly": round(kelly_val, 6),
        "value_index": vi,
        "tag": tag,
        "reasons": tuple(),
    }

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
        num = ev
        den = (o - 1.0)
        f = num/den if den>0 else None
        if f is None:
            return round(ev,6), None
        if f <= 0:
            return (round(ev,6), 0.0) if abs(f) <= 1e-9 else (round(ev,6), None)
        f_adj = min(1.0, max(0.0, f*k_fraction))
        if f_adj <= 0:
            return round(ev,6), 0.0
        return round(ev,6), round(f_adj, 6)
    except Exception:
        return None, None

# <<<M37_FILE: value_engine.py
