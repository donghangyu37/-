# >>>M37_FILE: value_engine.py
# value_engine.py
# 37号 · Value Engine —— 多模型融合 + 市场先验 + 四分盘 + 真实EV/Kelly + 质量工具

import math
try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - fallback for test environment
    class _RandomStub:
        @staticmethod
        def seed(_seed):
            return None

    class _NumpyStub:
        random = _RandomStub()

    np = _NumpyStub()  # type: ignore
from typing import Any, Dict, Tuple

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
            "quality_review": 0.6,
            "quality_reject": 0.3,
            "quality_target_bookmakers": 10,
            "consensus_alpha": 0.2,
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
            "quality_review": 0.6,
            "quality_reject": 0.3,
            "quality_target_bookmakers": 10,
            "consensus_alpha": 0.2,
        },
        "ah": {
            "keep_min": 0.02,
            "keep_max": 0.06,
            "drop": 0.12,
            "overround_range": (1.00, 1.25),
            "min_bookmakers": 6,
            "max_update_min": 10.0,
            "high_ev_review": 0.15,
            "high_ev_min_bookmakers": 8,
            "high_ev_max_update": 5.0,
            "high_ev_min_vi": 0.02,
            "quality_review": 0.6,
            "quality_reject": 0.3,
            "quality_target_bookmakers": 10,
            "consensus_alpha": 0.2,
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
            "quality_review": 0.6,
            "quality_reject": 0.3,
            "quality_target_bookmakers": 10,
            "consensus_alpha": 0.25,
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
            "quality_review": 0.6,
            "quality_reject": 0.3,
            "quality_target_bookmakers": 12,
            "consensus_alpha": 0.6,
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
            "quality_review": 0.6,
            "quality_reject": 0.3,
            "quality_target_bookmakers": 12,
            "consensus_alpha": 0.6,
        },
        "ah": {
            "keep_min": 0.04,
            "keep_max": 0.10,
            "drop": 0.18,
            "overround_range": (1.00, 1.25),
            "min_bookmakers": 8,
            "max_update_min": 10.0,
            "high_ev_review": 0.15,
            "high_ev_min_bookmakers": 8,
            "high_ev_max_update": 5.0,
            "high_ev_min_vi": 0.02,
            "quality_review": 0.6,
            "quality_reject": 0.3,
            "quality_target_bookmakers": 12,
            "consensus_alpha": 0.6,
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
            "quality_review": 0.6,
            "quality_reject": 0.3,
            "quality_target_bookmakers": 12,
            "consensus_alpha": 0.65,
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


def _clamp01(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _score_bookmakers(count: float | None, minimum: int, target: float | None) -> float:
    if count is None:
        return 0.6
    try:
        cnt = float(count)
    except (TypeError, ValueError):
        return 0.6
    min_req = max(1.0, float(minimum))
    if cnt <= min_req:
        return 0.65
    tgt = float(target) if target and float(target) > min_req else min_req + 4.0
    span = max(tgt - min_req, 1.0)
    return 0.65 + 0.35 * _clamp01((cnt - min_req) / span)


def _score_overround(overround: float | None, lo: float, hi: float) -> float:
    if overround is None:
        return 0.6
    try:
        val = float(overround)
    except (TypeError, ValueError):
        return 0.6
    if val < lo or val > hi:
        return 0.0
    mid = (lo + hi) / 2.0
    half_span = max((hi - lo) / 2.0, 1e-6)
    dist = abs(val - mid)
    ratio = min(1.0, dist / half_span)
    return 0.6 + 0.4 * (1.0 - ratio)


def _score_update_age(update_age: float | None, max_update: float) -> float:
    if update_age is None:
        return 0.75
    try:
        val = float(update_age)
    except (TypeError, ValueError):
        return 0.75
    if val <= 0:
        return 1.0
    max_allow = max(float(max_update), 1e-6)
    ratio = min(1.0, val / max_allow)
    return 0.5 + 0.5 * (1.0 - ratio)


def _score_data_quality(data_quality: float | None) -> float:
    if data_quality is None:
        return 0.7
    return 0.4 + 0.6 * _clamp01(data_quality)


def _score_sample_size(sample_size: float | None, target: float | None) -> float:
    if sample_size is None:
        return 0.7
    try:
        size = float(sample_size)
    except (TypeError, ValueError):
        return 0.7
    tgt = max(float(target) if target else 50.0, 1.0)
    return 0.5 + 0.5 * _clamp01(size / tgt)


def compute_market_quality(
    *,
    min_bookmakers: float | None,
    overround: float | None,
    update_age: float | None,
    policy: Dict[str, float],
    data_quality: float | None = None,
    sample_size: float | None = None,
) -> tuple[float, tuple[str, ...]]:
    min_req = int(policy.get("min_bookmakers", 6))
    target_books = policy.get("quality_target_bookmakers", min_req + 4)
    lo, hi = policy.get("overround_range", (1.02, 1.12))
    max_update = float(policy.get("max_update_min", 10.0))
    sample_target = policy.get("quality_sample_target")

    components: list[tuple[str, float]] = []
    components.append(("bookmakers", _score_bookmakers(min_bookmakers, min_req, target_books)))
    components.append(("overround", _score_overround(overround, float(lo), float(hi))))
    components.append(("stale", _score_update_age(update_age, max_update)))
    if data_quality is not None:
        components.append(("data", _score_data_quality(data_quality)))
    if sample_size is not None:
        components.append(("sample", _score_sample_size(sample_size, sample_target)))

    if not components:
        return 1.0, tuple()

    quality = sum(score for _, score in components) / len(components)
    flags = tuple(name for name, score in components if score < 0.55)
    return _clamp01(quality), flags


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
    odds: float | None = None,
    model_probability: float | None = None,
    consensus_probability: float | None = None,
    data_quality: float | None = None,
    sample_size: float | None = None,
) -> Dict[str, object]:
    result: Dict[str, Any] = {
        "ev": None,
        "kelly": None,
        "value_index": None,
        "tag": "invalid",
        "reasons": tuple(),
        "quality": None,
        "ev_input": None,
        "ev_calibrated": None,
        "thresholds": {},
    }
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

    def _to_float(value) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    min_req = int(policy.get("min_bookmakers", 6 if tier == LEAGUE_TIER_TOP else 8))
    try:
        or_range = policy.get("overround_range", (1.02, 1.12))
        lo, hi = float(or_range[0]), float(or_range[1])
    except Exception:
        lo, hi = 1.02, 1.12
    max_update = float(policy.get("max_update_min", 10.0))
    quality_review_threshold = float(policy.get("quality_review", 0.6))
    quality_reject_threshold = float(policy.get("quality_reject", 0.3))
    consensus_alpha = float(policy.get("consensus_alpha", 0.2 if tier == LEAGUE_TIER_TOP else 0.6))

    thresholds = {
        "keep_min": keep_min,
        "keep_max": keep_max,
        "drop": drop,
        "min_bookmakers": float(min_req),
        "overround_min": lo,
        "overround_max": hi,
        "max_update_min": max_update,
        "quality_review": quality_review_threshold,
        "quality_reject": quality_reject_threshold,
        "consensus_alpha": consensus_alpha,
        "kelly_cap": cap,
    }
    high_ev_review = _to_float(policy.get("high_ev_review"))
    if high_ev_review is not None:
        thresholds["high_ev_review"] = high_ev_review
    high_ev_min_books = _to_float(policy.get("high_ev_min_bookmakers"))
    if high_ev_min_books is not None:
        thresholds["high_ev_min_bookmakers"] = high_ev_min_books
    high_ev_max_update = _to_float(policy.get("high_ev_max_update"))
    if high_ev_max_update is not None:
        thresholds["high_ev_max_update"] = high_ev_max_update
    high_ev_min_vi = _to_float(policy.get("high_ev_min_vi"))
    if high_ev_min_vi is not None:
        thresholds["high_ev_min_vi"] = high_ev_min_vi

    result["thresholds"] = thresholds
    result["ev_input"] = ev_val

    if ev_val > drop:
        reason = f"ev>{drop:.1%}"
        result.update({"tag": "reject", "reasons": (reason,)})
        return result

    if ev_val < keep_min:
        tag = "low"
    elif ev_val <= keep_max:
        tag = "keep"
    else:
        tag = "review"

    bookmaker_count = _to_float(min_bookmakers)
    missing_liquidity = False
    low_liquidity = False
    if bookmaker_count is None or bookmaker_count <= 0:
        missing_liquidity = True
    elif bookmaker_count < float(min_req):
        low_liquidity = True

    reasons = []
    if missing_liquidity:
        reasons.append("bookmakers")

    try:
        overround_val = float(overround) if overround is not None else None
    except (TypeError, ValueError):
        overround_val = None
    if overround_val is not None and not (lo <= overround_val <= hi):
        reasons.append("overround")

    try:
        upd = float(update_age) if update_age is not None else None
    except (TypeError, ValueError):
        upd = None
    if upd is not None and upd > max_update:
        reasons.append("stale")

    quality, quality_flags = compute_market_quality(
        min_bookmakers=bookmaker_count,
        overround=overround_val,
        update_age=upd,
        policy=policy,
        data_quality=data_quality,
        sample_size=sample_size,
    )
    result["quality"] = quality

    if reasons:
        result.update({"tag": "reject", "reasons": tuple(reasons)})
        return result

    if quality <= quality_reject_threshold:
        reasons = tuple(sorted(set(quality_flags + ("quality",))))
        result.update({"tag": "reject", "reasons": reasons})
        return result

    ev_calibrated = ev_val
    odds_val = None
    try:
        if odds is not None:
            odds_val = float(odds)
            if odds_val <= 1.0:
                odds_val = None
    except (TypeError, ValueError):
        odds_val = None

    model_p = _clamp01(model_probability) if model_probability is not None else None
    consensus_p = _clamp01(consensus_probability) if consensus_probability is not None else None

    if odds_val is not None and model_p is not None:
        blend_p = model_p
        if consensus_p is not None:
            blend_p = _clamp01((1.0 - consensus_alpha) * model_p + consensus_alpha * consensus_p)
            ev_consensus = consensus_p * odds_val - 1.0
            ev_candidate = (1.0 - consensus_alpha) * ev_val + consensus_alpha * ev_consensus
            ev_calibrated = min(ev_val, ev_candidate)
        b = odds_val - 1.0
        if b > 0:
            kelly_from_blend = max(0.0, (blend_p * odds_val - 1.0) / b)
            kelly_val = min(kelly_val, kelly_from_blend)
    elif odds_val is not None and consensus_p is not None:
        ev_consensus = consensus_p * odds_val - 1.0
        ev_calibrated = min(ev_val, (1.0 - consensus_alpha) * ev_val + consensus_alpha * ev_consensus)

    result["ev_calibrated"] = ev_calibrated

    ev_penalized = ev_calibrated * quality
    kelly_val = max(0.0, min(kelly_val, kelly_val * quality))

    if kelly_val <= 0:
        result.update({"tag": "reject", "reasons": ("kelly",)})
        return result

    if ev_penalized > drop:
        reason = f"ev>{drop:.1%}"
        result.update({"tag": "reject", "reasons": (reason,)})
        return result

    if ev_penalized < keep_min:
        tag = "low"
    elif ev_penalized <= keep_max:
        tag = "keep"
    else:
        tag = "review"

    if quality < quality_review_threshold and tag == "keep":
        tag = "review"
    if low_liquidity and tag == "keep":
        tag = "review"

    vi = value_index(ev_penalized, kelly_val)
    high_ev_cut = policy.get("high_ev_review")
    if high_ev_cut is not None and ev_penalized > float(high_ev_cut):
        guard_reasons = []
        guard_books_val = policy.get("high_ev_min_bookmakers", min_req)
        try:
            guard_books = float(guard_books_val)
        except (TypeError, ValueError):
            guard_books = float(min_req)
        if bookmaker_count is None or bookmaker_count < guard_books:
            guard_reasons.append("hi_ev_books")
        guard_max_update = float(policy.get("high_ev_max_update", max_update))
        if upd is None or upd > guard_max_update:
            guard_reasons.append("hi_ev_stale")
        guard_vi = float(policy.get("high_ev_min_vi", 0.02))
        if vi is None or vi < guard_vi:
            guard_reasons.append("hi_ev_vi")
        if guard_reasons:
            result.update({"tag": "reject", "reasons": tuple(guard_reasons)})
            return result

    result.update(
        {
            "ev": round(ev_penalized, 6),
            "kelly": round(kelly_val, 6),
            "value_index": vi,
            "tag": tag,
            "reasons": tuple(),
            "quality": quality,
            "ev_input": ev_val,
            "ev_calibrated": ev_calibrated,
        }
    )
    return result

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
