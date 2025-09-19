import math

from daily_.value_engine import LEAGUE_TIER_OTHER, LEAGUE_TIER_TOP, evaluate_ev_market


def test_evaluate_rejects_insufficient_bookmakers() -> None:
    res = evaluate_ev_market(
        ev=0.05,
        kelly=0.05,
        market="ou",
        tier=LEAGUE_TIER_TOP,
        min_bookmakers=4,
        overround=1.05,
        update_age=5,
        odds=2.0,
        model_probability=0.55,
    )
    assert res["ev"] is None
    assert res["tag"] == "invalid"
    assert "bookmakers" in res.get("reasons", ())
    assert res.get("liquidity") is not None


def test_consensus_shrinks_ev_for_top_tier() -> None:
    res = evaluate_ev_market(
        ev=0.12,
        kelly=0.08,
        market="ou",
        tier=LEAGUE_TIER_TOP,
        min_bookmakers=10,
        overround=1.06,
        update_age=4,
        odds=2.1,
        model_probability=0.58,
        consensus_probability=0.52,
    )
    assert res["ev"] is not None
    assert math.isclose(res["ev_input"], 0.12, rel_tol=1e-9)
    assert res["ev"] < res["ev_input"]
    assert res["quality"] is not None and 0 < res["quality"] <= 1
    assert res.get("score") is not None
    assert res.get("liquidity") is not None


def test_quality_penalty_reduces_ev() -> None:
    res = evaluate_ev_market(
        ev=0.08,
        kelly=0.09,
        market="ou",
        tier=LEAGUE_TIER_OTHER,
        min_bookmakers=8,
        overround=1.12,
        update_age=9.5,
        odds=2.05,
        model_probability=0.57,
        consensus_probability=0.5,
        data_quality=0.4,
        sample_size=6,
    )
    assert res["ev_calibrated"] is not None
    assert res["quality"] is not None and res["quality"] < 1.0
    assert res["ev_calibrated"] <= res["ev_input"]
    assert res["tag"] in {"reject", "invalid"}
    assert "quality" in (res.get("reasons") or ())


def test_scoring_retains_high_quality_market() -> None:
    res = evaluate_ev_market(
        ev=0.06,
        kelly=0.06,
        market="ou",
        tier=LEAGUE_TIER_TOP,
        min_bookmakers=10,
        overround=1.06,
        update_age=3,
        odds=2.05,
        model_probability=0.56,
        consensus_probability=0.54,
        data_quality=0.95,
        sample_size=24,
    )
    assert res["tag"] in {"keep", "review"}
    assert res.get("score") is not None and res["score"] > 0
    assert res.get("kelly_full") is not None and res["kelly_full"] > 0
    assert res.get("liquidity") is not None and res["liquidity"] >= 0.9
    assert res.get("threshold_score_keep") is not None
    assert math.isclose(res["threshold_score_keep"], 0.06, rel_tol=1e-9)
    assert math.isclose(res["threshold_quality_keep"], 0.78, rel_tol=1e-9)
    assert math.isclose(res["threshold_liquidity_keep"], 0.9, rel_tol=1e-9)
    assert math.isclose(res["threshold_kelly_keep"], 0.02, rel_tol=1e-9)
    assert math.isclose(res["threshold_ev_keep_min"], 0.02, rel_tol=1e-9)


def test_low_quality_market_fails_gate() -> None:
    res = evaluate_ev_market(
        ev=0.08,
        kelly=0.05,
        market="ou",
        tier=LEAGUE_TIER_OTHER,
        min_bookmakers=9,
        overround=1.11,
        update_age=9,
        odds=2.1,
        model_probability=0.58,
        consensus_probability=0.57,
        data_quality=0.2,
        sample_size=2,
    )
    assert res["tag"] in {"reject", "invalid"}
    assert "quality" in (res.get("reasons") or ())
