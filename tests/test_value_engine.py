import math

from daily_.value_engine import LEAGUE_TIER_OTHER, LEAGUE_TIER_TOP, evaluate_ev_market


def test_evaluate_softens_low_bookmaker_penalty() -> None:
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
    assert res["ev"] is not None
    assert res["kelly"] is not None
    assert res["tag"] == "review"
    assert "bookmakers" not in res.get("reasons", ())
    assert math.isclose(res["ev"], 0.037333, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(res["kelly"], 0.037333, rel_tol=1e-9, abs_tol=1e-9)
    thresholds = res.get("thresholds")
    assert isinstance(thresholds, dict)
    assert math.isclose(float(thresholds["keep_min"]), 0.02, rel_tol=1e-9)
    assert math.isclose(float(thresholds["keep_max"]), 0.06, rel_tol=1e-9)
    assert math.isclose(float(thresholds["drop"]), 0.12, rel_tol=1e-9)
    assert math.isclose(float(thresholds["min_bookmakers"]), 6.0, rel_tol=1e-9)


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
    thresholds = res.get("thresholds")
    assert isinstance(thresholds, dict)
    assert math.isclose(float(thresholds["consensus_alpha"]), 0.2, rel_tol=1e-9)
    assert math.isclose(float(thresholds["kelly_cap"]), 0.08, rel_tol=1e-9)


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
    assert res["ev"] is not None
    assert res["ev_calibrated"] is not None
    assert res["quality"] is not None and res["quality"] < 1.0
    assert res["ev"] < res["ev_calibrated"]
    thresholds = res.get("thresholds")
    assert isinstance(thresholds, dict)
    assert math.isclose(float(thresholds["quality_reject"]), 0.3, rel_tol=1e-9)
    assert math.isclose(float(thresholds["quality_review"]), 0.6, rel_tol=1e-9)
