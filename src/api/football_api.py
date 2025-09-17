# === src/api/football_api.py · 并发复用 + 退避重试 + 限速 + 轻量缓存 + 主盘/全线聚合（含角球/Asian Totals） ===
from __future__ import annotations

import os, json, time, hashlib, threading, re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

# ===== 环境变量与基础配置 =====
load_dotenv()
API_KEY = (
    os.getenv("FOOTBALL_API_KEY")
    or os.getenv("API_FOOTBALL_KEY")
    or os.getenv("X_APISPORTS_KEY")
    or os.getenv("API_KEY")
)
if not API_KEY:
    raise RuntimeError("未找到 API Key，请在 .env 中设置（FOOTBALL_API_KEY / API_FOOTBALL_KEY / X_APISPORTS_KEY / API_KEY 任一）")

BASE_URL = os.getenv("FOOTBALL_API_BASE", "https://v3.football.api-sports.io").rstrip("/")
HEADERS  = {"x-apisports-key": API_KEY}

USE_PROXY = os.getenv("USE_PROXY", "0") == "1"
RATE_QPS = float(os.getenv("RATE_QPS", "3.0"))
_MIN_INTERVAL = 1.0 / max(0.1, RATE_QPS)
_last_call_ts = 0.0

_SESSION = requests.Session()
_SESSION.trust_env = USE_PROXY
_RETRY = Retry(
    total=5, connect=5, read=5,
    backoff_factor=1.0,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset({"GET"}),
    raise_on_status=False,
)
_ADAPTER = HTTPAdapter(max_retries=_RETRY, pool_connections=50, pool_maxsize=50)
_SESSION.mount("https://", _ADAPTER)
_SESSION.mount("http://", _ADAPTER)
_SESSION_LOCK = threading.Lock()

_MEMO: Dict[str, tuple[float, dict]] = {}

def _memo_key(endpoint: str, params: dict) -> str:
    s = json.dumps({"e": endpoint, "p": params}, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def _get(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    global _last_call_ts
    now = time.time()
    wait = _MIN_INTERVAL - (now - _last_call_ts)
    if wait > 0:
        time.sleep(wait)

    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    try:
        with _SESSION_LOCK:
            r = _SESSION.get(url, headers=HEADERS, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"网络请求失败: {e}") from e
    finally:
        _last_call_ts = time.time()

    if isinstance(data, dict) and data.get("errors"):
        raise RuntimeError(f"API返回错误: {data['errors']}")
    return data if isinstance(data, dict) else {}

def _get_cached(endpoint: str, params: dict, ttl_sec: int) -> dict:
    key = _memo_key(endpoint, params)
    now = time.time()
    ent = _MEMO.get(key)
    if ent and ent[0] > now:
        return ent[1]
    data = _get(endpoint, params)
    _MEMO[key] = (now + ttl_sec, data)
    return data

# ===== 业务封装 =====
def fixtures_by_date(date_str: str) -> List[Dict]:
    data = _get_cached("fixtures", {"date": date_str}, ttl_sec=120)
    return data.get("response", [])

def list_teams_in_league(league_id: int, season: int) -> List[Dict]:
    params = {"league": league_id, "season": season}
    data = _get_cached("teams", params, ttl_sec=24*3600)
    return data.get("response", [])

def team_statistics(league_id: int, season: int, team_id: int) -> Dict:
    params = {"league": league_id, "season": season, "team": team_id}
    data = _get_cached("teams/statistics", params, ttl_sec=12*3600)
    return data.get("response", {})

def recent_fixtures_by_team(team_id: int, season: int, last: int = 20, league_id: Optional[int] = None) -> List[Dict]:
    params: Dict[str, Any] = {"team": team_id, "season": season, "last": last}
    if league_id:
        params["league"] = league_id
    data = _get_cached("fixtures", params, ttl_sec=3600)
    return data.get("response", [])

def find_league(country: str, league_name: str, season: int) -> Optional[int]:
    data = _get_cached("leagues", {"country": country, "name": league_name, "season": season}, ttl_sec=24*3600)
    for it in data.get("response", []):
        name = (it.get("league", {}) or {}).get("name", "")
        if str(name).lower() == str(league_name).lower():
            return it.get("league", {}).get("id")
    if data.get("response"):
        return data["response"][0]["league"]["id"]
    return None

_SIDE_TOKENS = {
    "home": "home",
    "1": "home",
    "team1": "home",
    "h": "home",
    "away": "away",
    "2": "away",
    "team2": "away",
    "a": "away",
    "guest": "away",
    "visitor": "away",
}

_PICK_TOKENS = {"pk", "pick", "pick'em", "pickem", "p.k.", "p.k"}
_NUM_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")


def _parse_update_timestamp(value) -> float | None:
    """Parse bookmaker update timestamps into epoch seconds."""

    if value is None:
        return None

    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    text = str(value).strip()
    if not text:
        return None

    # Normalise common ISO8601 variations returned by the API.
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.timestamp()


def _median(arr):
    arr = [x for x in (arr or []) if x]
    if not arr:
        return None
    arr = sorted(arr)
    n = len(arr)
    return float(arr[n // 2]) if n % 2 == 1 else float((arr[n // 2 - 1] + arr[n // 2]) / 2)


def _parse_float(value, default=None):
    if value is None:
        return default
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    text = str(value).strip()
    if not text:
        return default
    text_norm = text.lower().replace("−", "-")
    if text_norm in _PICK_TOKENS:
        return 0.0

    try:
        return float(text_norm.replace(",", "."))
    except ValueError:
        match = _NUM_RE.search(text_norm)
        if match:
            token = match.group(0).replace(",", ".")
            try:
                return float(token)
            except ValueError:
                return default
    return default


def _normalise_side(value) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    text = text.replace("home team", "home").replace("away team", "away")
    if "home" in text:
        return "home"
    if "away" in text or "visitor" in text or "guest" in text:
        return "away"

    tokens = [tok for tok in re.split(r"[^a-z0-9]+", text) if tok]
    for token in tokens:
        if token in _SIDE_TOKENS:
            return _SIDE_TOKENS[token]
    return None


def _extract_handicap(entry: Dict[str, Any]) -> float | None:
    line = _parse_float(entry.get("handicap"))
    if line is not None:
        return line
    return _parse_float(entry.get("value"))

def odds_by_fixture(fixture_id: int) -> Dict[str, float]:
    try:
        data = _get_cached("odds", {"fixture": fixture_id}, ttl_sec=180)
    except Exception:
        return {}

    def is_goal_ou(name: str) -> bool:
        n = (name or "").lower()
        # 常见命名 + Asian Total Goals / Goal Line
        goal_keys = [
            "over/under", "over under", "totals", "total goals",
            "goals over/under", "goal line", "asian total", "asian totals",
            "asian over/under", "o/u"
        ]
        if any(k in n for k in goal_keys) or ("over" in n or "under" in n):
            forbid = ["1st half","first half","2nd half","second half","half","team","corners","cards","booking"]
            return not any(t in n for t in forbid)
        return False

    def is_corners_ou(name: str) -> bool:
        n = (name or "").lower()
        if "corner" in n and ("over" in n or "under" in n or "total" in n):
            forbid = ["1st half","first half","2nd half","second half","half","team","cards","booking"]
            return not any(t in n for t in forbid)
        return False

    def is_asian_handicap(name: str) -> bool:
        n = (name or "").lower()
        if "asian handicap" in n or ("handicap" in n and "european" not in n):
            forbid = [
                "1st half",
                "first half",
                "2nd half",
                "second half",
                "half time",
                "half-time",
                "halftime",
            ]
            if any(term in n for term in forbid):
                return False
            return "corner" not in n and "cards" not in n and "booking" not in n
        return False

    def _parse_float(x):
        try:
            return float(str(x).replace(",", "."))
        except Exception:
            return None

    def _add_safe(lst, odd):
        try:
            v = float(odd)
            if 1.10 <= v <= 50.0:
                lst.append(v)
        except Exception:
            pass

    h_list, d_list, a_list = [], [], []
    ou_map: Dict[float, Dict[str, List[float]]] = {}
    ah_map: Dict[float, Dict[str, List[float]]] = {}   # home_line -> {"home": [], "away": []}
    cr_map: Dict[float, Dict[str, List[float]]] = {}

    ou_updates: dict[float, List[float]] = defaultdict(list)
    ah_updates: dict[float, List[float]] = defaultdict(list)
    cr_updates: dict[float, List[float]] = defaultdict(list)
    one_x_two_updates: List[float] = []

    # —— 解析各博彩公司盘口 —— #
    for item in data.get("response", []):
        for bk in item.get("bookmakers") or []:
            update_ts = _parse_update_timestamp(
                bk.get("last_update")
                or bk.get("lastUpdate")
                or bk.get("update")
            )

            for b in bk.get("bets") or []:
                name = b.get("name") or ""
                values = b.get("values") or []

                # 1X2
                if "match winner" in name.lower() or name.lower() in ("1x2","match odds","winner"):
                    used_market = False
                    for v in values:
                        val = (v.get("value") or "").lower().strip()
                        odd = v.get("odd")
                        if val in ("home","1"):
                            _add_safe(h_list, odd)
                            used_market = True
                        elif val in ("draw","x"):
                            _add_safe(d_list, odd)
                            used_market = True
                        elif val in ("away","2"):
                            _add_safe(a_list, odd)
                            used_market = True
                    if used_market and update_ts is not None:
                        one_x_two_updates.append(update_ts)

                # 进球 OU（含 Asian Total / Goal Line）
                if is_goal_ou(name):
                    lines_used: set[float] = set()
                    for v in values:
                        side_txt = (v.get("value") or "").lower().strip()
                        odd = v.get("odd")
                        line = _parse_float(v.get("handicap"))
                        if line is None:
                            s = side_txt.replace(" ", "")
                            if s.startswith("over"):  line = _parse_float(s[4:])
                            elif s.startswith("under"): line = _parse_float(s[5:])
                        if line is None:
                            continue
                        line_f = float(line)
                        if side_txt.startswith("over"):
                            ou_map.setdefault(line_f, {"over": [], "under": []})["over"].append(_parse_float(odd) or 0.0)
                            lines_used.add(line_f)
                        elif side_txt.startswith("under"):
                            ou_map.setdefault(line_f, {"over": [], "under": []})["under"].append(_parse_float(odd) or 0.0)
                            lines_used.add(line_f)
                    if update_ts is not None:
                        for ln in lines_used:
                            ou_updates[ln].append(update_ts)

                # 角球 OU
                if is_corners_ou(name):
                    lines_used: set[float] = set()
                    for v in values:
                        side_txt = (v.get("value") or "").lower().strip()
                        odd = v.get("odd")
                        line = _parse_float(v.get("handicap"))
                        if line is None:
                            s = side_txt.replace(" ", "")
                            if s.startswith("over"):  line = _parse_float(s[4:])
                            elif s.startswith("under"): line = _parse_float(s[5:])
                        if line is None:
                            continue
                        line_f = float(line)
                        if side_txt.startswith("over"):
                            cr_map.setdefault(line_f, {"over": [], "under": []})["over"].append(_parse_float(odd) or 0.0)
                            lines_used.add(line_f)
                        elif side_txt.startswith("under"):
                            cr_map.setdefault(line_f, {"over": [], "under": []})["under"].append(_parse_float(odd) or 0.0)
                            lines_used.add(line_f)
                    if update_ts is not None:
                        for ln in lines_used:
                            cr_updates[ln].append(update_ts)

                # 亚洲让球（非角球）
                if is_asian_handicap(name):
                    lines_used: set[float] = set()
                    for v in values:
                        side_txt = _normalise_side(v.get("value"))
                        odd = _parse_float(v.get("odd"))
                        line = _extract_handicap(v)
                        if side_txt not in ("home", "away") or line is None or odd is None:
                            continue
                        if not (1.10 <= odd <= 50.0):
                            continue

                        line_val = float(line)
                        # Reuse existing keys so that Home/Away selections from the same
                        # bookmaker line land in the same entry (处理正负盘符). This fixes
                        # cases where one side was inserted first with the opposite sign.
                        resolved_line: Optional[float] = None
                        for cand in (line_val, -line_val):
                            if cand in ah_map:
                                resolved_line = cand
                                break
                        if resolved_line is None:
                            resolved_line = line_val if side_txt == "home" else -line_val

                        ah_map.setdefault(resolved_line, {"home": [], "away": []})
                        ah_map[resolved_line][side_txt].append(odd)
                        lines_used.add(resolved_line)
                    if update_ts is not None:
                        for ln in lines_used:
                            ah_updates[ln].append(update_ts)

    out: Dict[str, float] = {}

    # —— 1X2 主盘中位
    mh, md, ma = _median(h_list), _median(d_list), _median(a_list)
    if mh and md and ma:
        imp_sum = 1.0/mh + 1.0/md + 1.0/ma
        if 1.02 <= imp_sum <= 1.30 and min(len(h_list),len(d_list),len(a_list)) >= 2:
            out["1x2_home"], out["1x2_draw"], out["1x2_away"] = mh, md, ma
            out["1x2_overround"] = float(imp_sum)
            out["1x2_home_cnt"], out["1x2_draw_cnt"], out["1x2_away_cnt"] = int(len(h_list)), int(len(d_list)), int(len(a_list))
            if one_x_two_updates:
                out["1x2_last_update_ts"] = max(one_x_two_updates)

    # —— Goals OU 主盘
    main = None
    for line, sides in ou_map.items():
        co = len([x for x in sides.get("over", []) if x])
        cu = len([x for x in sides.get("under", []) if x])
        if co >= 2 and cu >= 2:
            om = _median(sides["over"]); um = _median(sides["under"])
            if not om or not um:
                continue
            oround = 1.0/om + 1.0/um
            if not (1.02 <= oround <= 1.18):
                continue
            cand = (co+cu, -abs(1.0/om - 1.0/um), float(line), om, um, co, cu, oround)
            if (main is None) or (cand > main):
                main = cand
    if main:
        _,_,line,om,um,co,cu,oround = main
        out["ou_main_line"] = float(line)
        out["ou_main_over"], out["ou_main_under"] = float(om), float(um)
        out["ou_main_over_cnt"], out["ou_main_under_cnt"], out["ou_main_overround"] = int(co), int(cu), float(oround)
        ts_list = ou_updates.get(float(line))
        if ts_list:
            out["ou_main_last_update_ts"] = max(ts_list)

    # —— Goals OU@2.5 参考
    if 2.5 in ou_map:
        sides = ou_map[2.5]
        co = len([x for x in sides.get("over", []) if x]); cu = len([x for x in sides.get("under", []) if x])
        if co >= 2 and cu >= 2:
            om = _median(sides["over"]); um = _median(sides["under"])
            if om and um:
                out["ou_over_2_5"], out["ou_under_2_5"] = float(om), float(um)

    # —— AH 主盘（较宽松）
    pick = None
    for line, sides in ah_map.items():
        hs, as_ = sides.get("home", []), sides.get("away", [])
        ch, ca = len(hs), len(as_)
        if ch >= 1 and ca >= 1:
            oh, oa = _median(hs), _median(as_)
            if not oh or not oa:
                continue
            oround = 1.0/oh + 1.0/oa
            if not (1.00 <= oround <= 1.25):
                continue
            cand = (abs(line), abs(oh - oa), float(line), oh, oa, ch, ca, oround)
            if (pick is None) or (cand < pick):
                pick = cand
    if pick:
        _,_,line,oh,oa,ch,ca,oround = pick
        out["ah_line"], out["ah_home_odds"], out["ah_away_odds"] = float(line), float(oh), float(oa)
        out["ah_home_cnt"], out["ah_away_cnt"], out["ah_overround"] = int(ch), int(ca), float(oround)
        ts_list = ah_updates.get(float(line))
        if ts_list:
            out["ah_last_update_ts"] = max(ts_list)

    # —— 角球 OU 主盘
    cmain = None
    for line, sides in cr_map.items():
        co = len([x for x in sides.get("over", []) if x]); cu = len([x for x in sides.get("under", []) if x])
        if co >= 2 and cu >= 2:
            om = _median(sides["over"]); um = _median(sides["under"])
            if not om or not um:
                continue
            oround = 1.0/om + 1.0/um
            if not (1.02 <= oround <= 1.20):
                continue
            cand = (co+cu, -abs(1.0/om - 1.0/um), float(line), om, um, co, cu, oround)
            if (cmain is None) or (cand > cmain):
                cmain = cand
    if cmain:
        _,_,line,om,um,co,cu,oround = cmain
        out["crn_main_line"] = float(line)
        out["crn_main_over"], out["crn_main_under"] = float(om), float(um)
        out["crn_main_over_cnt"], out["crn_main_under_cnt"], out["crn_main_overround"] = int(co), int(cu), float(oround)
        ts_list = cr_updates.get(float(line))
        if ts_list:
            out["crn_main_last_update_ts"] = max(ts_list)

    # —— 全线快照（用于“全线 Top 10”打印） —— #
    ou_all_list = []
    for line, sides in ou_map.items():
        ov, un = [x for x in sides.get("over", []) if x], [x for x in sides.get("under", []) if x]
        if len(ov) >= 1 and len(un) >= 1:
            om, um = _median(ov), _median(un)
            if not om or not um:
                continue
            oround = 1.0/om + 1.0/um
            if 1.00 <= oround <= 1.25:
                ou_all_list.append({"line": float(line), "over": float(om), "under": float(um),
                                    "co": int(len(ov)), "cu": int(len(un)), "overround": float(oround)})
    if ou_all_list:
        out["ou_all"] = sorted(ou_all_list, key=lambda x: x["line"])

    crn_all_list = []
    for line, sides in cr_map.items():
        ov, un = [x for x in sides.get("over", []) if x], [x for x in sides.get("under", []) if x]
        if len(ov) >= 1 and len(un) >= 1:
            om, um = _median(ov), _median(un)
            if not om or not um:
                continue
            oround = 1.0/om + 1.0/um
            if 1.00 <= oround <= 1.30:
                crn_all_list.append({"line": float(line), "over": float(om), "under": float(um),
                                     "co": int(len(ov)), "cu": int(len(un)), "overround": float(oround)})
    if crn_all_list:
        out["crn_all"] = sorted(crn_all_list, key=lambda x: x["line"])

    # —— 原始盘口映射（供“盘口矩阵”导出） —— #
    def _clean_ou_or_cr_map(m):
        outm = {}
        for line, sides in m.items():
            try:
                key = float(line)
            except Exception:
                continue
            over = [float(x) for x in sides.get("over", []) if x]
            under = [float(x) for x in sides.get("under", []) if x]
            outm[key] = {"over": over, "under": under}
        return outm

    def _clean_ah_map(m):
        outm = {}
        for line, sides in m.items():
            try:
                key = float(line)
            except Exception:
                continue
            home = [float(x) for x in sides.get("home", []) if x]
            away = [float(x) for x in sides.get("away", []) if x]
            outm[key] = {"home": home, "away": away}
        return outm

    clean_ou_map = _clean_ou_or_cr_map(ou_map)
    clean_cr_map = _clean_ou_or_cr_map(cr_map)
    clean_ah_map = _clean_ah_map(ah_map)

    out["_raw_ou_map"]  = clean_ou_map
    out["_raw_crn_map"] = clean_cr_map
    out["_raw_ah_map"]  = clean_ah_map

    # —— 已清洗“全线矩阵”摘要（中位数、样本、超额） —— #
    def pack_ou_lines(src: Dict[float, Dict[str, List[float]]], updates: dict[float, List[float]]) -> Dict[str, Dict[str, float]]:
        res: Dict[str, Dict[str, float]] = {}
        for line, sides in src.items():
            ov = [x for x in sides.get("over", []) if x]
            un = [x for x in sides.get("under", []) if x]
            if not ov or not un:
                continue
            om, um = _median(ov), _median(un)
            if not om or not um:
                continue
            overround = 1.0/om + 1.0/um
            entry = {
                "over_median": float(om),
                "under_median": float(um),
                "over_cnt": int(len(ov)),
                "under_cnt": int(len(un)),
                "overround": float(overround),
            }
            ts_list = updates.get(float(line))
            if ts_list:
                entry["last_update_ts"] = max(ts_list)
            res[str(float(line))] = entry
        return res

    def pack_ah_lines(src: Dict[float, Dict[str, List[float]]], updates: dict[float, List[float]]) -> Dict[str, Dict[str, float]]:
        res: Dict[str, Dict[str, float]] = {}
        for line, sides in src.items():
            hs = [x for x in sides.get("home", []) if x]
            as_ = [x for x in sides.get("away", []) if x]
            if not hs or not as_:
                continue
            oh, oa = _median(hs), _median(as_)
            if not oh or not oa:
                continue
            overround = 1.0/oh + 1.0/oa
            entry = {
                "home_median": float(oh),
                "away_median": float(oa),
                "home_cnt": int(len(hs)),
                "away_cnt": int(len(as_)),
                "overround": float(overround),
            }
            ts_list = updates.get(float(line))
            if ts_list:
                entry["last_update_ts"] = max(ts_list)
            res[str(float(line))] = entry
        return res

    out["ou_lines"]  = pack_ou_lines(clean_ou_map, ou_updates)
    out["crn_lines"] = pack_ou_lines(clean_cr_map, cr_updates)
    out["ah_lines"]  = pack_ah_lines(clean_ah_map, ah_updates)

    return out

__all__ = [
    "fixtures_by_date", "list_teams_in_league", "team_statistics",
    "recent_fixtures_by_team", "find_league", "odds_by_fixture"
]
