# === src/api/football_api.py · 并发复用 + 退避重试 + 限速 + 轻量缓存 + 主盘/全线聚合（含角球/Asian Totals） ===
from __future__ import annotations

import os, json, time, hashlib, threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence
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


def _parse_update_minutes(value: Any, now_ts: float) -> Optional[float]:
    """Convert bookmaker update timestamps to "age" in minutes.

    The API may emit ISO8601 strings either with a trailing ``Z`` or explicit
    timezone offsets.  When parsing fails we return ``None`` so callers can
    decide how strictly to treat missing freshness information.
    """

    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None

    dt: Optional[datetime] = None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
    except ValueError:
        # Fallback formats observed occasionally without timezone markers.
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.strptime(text, fmt)
            except ValueError:
                continue
            else:
                dt = dt.replace(tzinfo=timezone.utc)
                break
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    age_minutes = max(0.0, (now_ts - dt.timestamp()) / 60.0)
    return float(age_minutes)

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

def _median(arr):
    arr = [x for x in (arr or []) if x]
    if not arr: return None
    arr = sorted(arr); n = len(arr)
    return float(arr[n // 2]) if n % 2 == 1 else float((arr[n // 2 - 1] + arr[n // 2]) / 2)

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

    now_ts = time.time()

    h_list, d_list, a_list = [], [], []
    h_bm: set[str] = set(); d_bm: set[str] = set(); a_bm: set[str] = set()
    h_updates: List[float] = []; d_updates: List[float] = []; a_updates: List[float] = []

    ou_map: Dict[float, Dict[str, Any]] = {}
    ah_map: Dict[float, Dict[str, Any]] = {}   # home_line -> {"home": [], "away": []}
    cr_map: Dict[float, Dict[str, Any]] = {}

    def _ou_bucket(dst: Dict[float, Dict[str, Any]], line_val: float) -> Dict[str, Any]:
        return dst.setdefault(
            float(line_val),
            {
                "over": [],
                "under": [],
                "_over_bm": set(),
                "_under_bm": set(),
                "_over_updates": [],
                "_under_updates": [],
            },
        )

    def _ah_bucket(line_val: float) -> Dict[str, Any]:
        return ah_map.setdefault(
            float(line_val),
            {
                "home": [],
                "away": [],
                "_home_bm": set(),
                "_away_bm": set(),
                "_home_updates": [],
                "_away_updates": [],
            },
        )

    # —— 解析各博彩公司盘口 —— #
    for item in data.get("response", []):
        for bk in item.get("bookmakers") or []:
            bk_id = bk.get("id")
            bk_name = bk.get("name")
            if bk_id is not None:
                bk_key = f"{bk_id}"
            elif bk_name:
                bk_key = str(bk_name)
            else:
                bk_key = f"bk_{id(bk)}"
            update_age = _parse_update_minutes(bk.get("update"), now_ts)

            for b in bk.get("bets") or []:
                name = b.get("name") or ""
                values = b.get("values") or []

                # 1X2
                if "match winner" in name.lower() or name.lower() in ("1x2","match odds","winner"):
                    for v in values:
                        val = (v.get("value") or "").lower().strip()
                        odd = v.get("odd")
                        if val in ("home","1"):
                            before = len(h_list)
                            _add_safe(h_list, odd)
                            if len(h_list) != before:
                                h_bm.add(bk_key)
                                if update_age is not None:
                                    h_updates.append(update_age)
                        elif val in ("draw","x"):
                            before = len(d_list)
                            _add_safe(d_list, odd)
                            if len(d_list) != before:
                                d_bm.add(bk_key)
                                if update_age is not None:
                                    d_updates.append(update_age)
                        elif val in ("away","2"):
                            before = len(a_list)
                            _add_safe(a_list, odd)
                            if len(a_list) != before:
                                a_bm.add(bk_key)
                                if update_age is not None:
                                    a_updates.append(update_age)

                # 进球 OU（含 Asian Total / Goal Line）
                if is_goal_ou(name):
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
                        bucket = _ou_bucket(ou_map, float(line))
                        val = _parse_float(odd)
                        if side_txt.startswith("over") and val:
                            bucket["over"].append(val)
                            bucket["_over_bm"].add(bk_key)
                            if update_age is not None:
                                bucket["_over_updates"].append(update_age)
                        elif side_txt.startswith("under") and val:
                            bucket["under"].append(val)
                            bucket["_under_bm"].add(bk_key)
                            if update_age is not None:
                                bucket["_under_updates"].append(update_age)

                # 角球 OU
                if is_corners_ou(name):
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
                        bucket = _ou_bucket(cr_map, float(line))
                        val = _parse_float(odd)
                        if side_txt.startswith("over") and val:
                            bucket["over"].append(val)
                            bucket["_over_bm"].add(bk_key)
                            if update_age is not None:
                                bucket["_over_updates"].append(update_age)
                        elif side_txt.startswith("under") and val:
                            bucket["under"].append(val)
                            bucket["_under_bm"].add(bk_key)
                            if update_age is not None:
                                bucket["_under_updates"].append(update_age)

                # 亚洲让球（非角球）
                if is_asian_handicap(name):
                    for v in values:
                        side_txt = (v.get("value") or "").lower().strip()   # "Home"/"Away"
                        odd = _parse_float(v.get("odd"))
                        line = _parse_float(v.get("handicap"))
                        if side_txt not in ("home","away") or line is None or odd is None:
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

                        bucket = _ah_bucket(resolved_line)
                        bucket[side_txt].append(odd)
                        bm_key = "_home_bm" if side_txt == "home" else "_away_bm"
                        upd_key = "_home_updates" if side_txt == "home" else "_away_updates"
                        bucket[bm_key].add(bk_key)
                        if update_age is not None:
                            bucket[upd_key].append(update_age)

    out: Dict[str, float] = {}

    # —— 1X2 主盘中位
    mh, md, ma = _median(h_list), _median(d_list), _median(a_list)
    cnt_h, cnt_d, cnt_a = len(h_bm), len(d_bm), len(a_bm)
    bm_count = min(cnt_h, cnt_d, cnt_a) if all(x > 0 for x in (cnt_h, cnt_d, cnt_a)) else 0
    max_update = None
    updates_combined: List[float] = [
        *([x for x in h_updates if x is not None]),
        *([x for x in d_updates if x is not None]),
        *([x for x in a_updates if x is not None]),
    ]
    if updates_combined:
        max_update = float(max(updates_combined))
    if mh and md and ma:
        imp_sum = 1.0/mh + 1.0/md + 1.0/ma
        if 1.02 <= imp_sum <= 1.30 and bm_count >= 2:
            out["1x2_home"], out["1x2_draw"], out["1x2_away"] = mh, md, ma
            out["1x2_overround"] = float(imp_sum)
            out["1x2_bookmaker_count"] = int(bm_count)
            if max_update is not None:
                out["1x2_max_update_minutes"] = max_update

    # —— Goals OU 主盘
    main = None
    for line, sides in ou_map.items():
        bm_over = len(sides.get("_over_bm", []))
        bm_under = len(sides.get("_under_bm", []))
        co = bm_over if bm_over else len([x for x in sides.get("over", []) if x])
        cu = bm_under if bm_under else len([x for x in sides.get("under", []) if x])
        if co >= 2 and cu >= 2:
            om = _median(sides["over"]); um = _median(sides["under"])
            if not om or not um:
                continue
            oround = 1.0/om + 1.0/um
            if not (1.02 <= oround <= 1.18):
                continue
            updates = [
                *[x for x in sides.get("_over_updates", []) if x is not None],
                *[x for x in sides.get("_under_updates", []) if x is not None],
            ]
            max_update = float(max(updates)) if updates else None
            cand = (co+cu, -abs(1.0/om - 1.0/um), float(line), om, um, co, cu, oround, max_update)
            if (main is None) or (cand > main):
                main = cand
    if main:
        *_, line, om, um, co, cu, oround, max_update = main
        out["ou_main_line"] = float(line)
        out["ou_main_over"], out["ou_main_under"] = float(om), float(um)
        out["ou_main_over_cnt"], out["ou_main_under_cnt"], out["ou_main_overround"] = int(co), int(cu), float(oround)
        out["ou_main_bookmaker_count"] = int(min(co, cu))
        if max_update is not None:
            out["ou_main_max_update_minutes"] = float(max_update)

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
        bm_home = len(sides.get("_home_bm", []))
        bm_away = len(sides.get("_away_bm", []))
        ch = bm_home if bm_home else len(hs)
        ca = bm_away if bm_away else len(as_)
        if ch >= 1 and ca >= 1:
            oh, oa = _median(hs), _median(as_)
            if not oh or not oa:
                continue
            oround = 1.0/oh + 1.0/oa
            if not (1.00 <= oround <= 1.25):
                continue
            updates = [
                *[x for x in sides.get("_home_updates", []) if x is not None],
                *[x for x in sides.get("_away_updates", []) if x is not None],
            ]
            max_update = float(max(updates)) if updates else None
            cand = (abs(line), abs(oh - oa), float(line), oh, oa, ch, ca, oround, max_update)
            if (pick is None) or (cand < pick):
                pick = cand
    if pick:
        *_, line, oh, oa, ch, ca, oround, max_update = pick
        out["ah_line"], out["ah_home_odds"], out["ah_away_odds"] = float(line), float(oh), float(oa)
        out["ah_home_cnt"], out["ah_away_cnt"], out["ah_overround"] = int(ch), int(ca), float(oround)
        out["ah_bookmaker_count"] = int(min(ch, ca))
        if max_update is not None:
            out["ah_max_update_minutes"] = float(max_update)

    # —— 角球 OU 主盘
    cmain = None
    for line, sides in cr_map.items():
        bm_over = len(sides.get("_over_bm", []))
        bm_under = len(sides.get("_under_bm", []))
        co = bm_over if bm_over else len([x for x in sides.get("over", []) if x])
        cu = bm_under if bm_under else len([x for x in sides.get("under", []) if x])
        if co >= 2 and cu >= 2:
            om = _median(sides["over"]); um = _median(sides["under"])
            if not om or not um:
                continue
            oround = 1.0/om + 1.0/um
            if not (1.02 <= oround <= 1.20):
                continue
            updates = [
                *[x for x in sides.get("_over_updates", []) if x is not None],
                *[x for x in sides.get("_under_updates", []) if x is not None],
            ]
            max_update = float(max(updates)) if updates else None
            cand = (co+cu, -abs(1.0/om - 1.0/um), float(line), om, um, co, cu, oround, max_update)
            if (cmain is None) or (cand > cmain):
                cmain = cand
    if cmain:
        *_, line, om, um, co, cu, oround, max_update = cmain
        out["crn_main_line"] = float(line)
        out["crn_main_over"], out["crn_main_under"] = float(om), float(um)
        out["crn_main_over_cnt"], out["crn_main_under_cnt"], out["crn_main_overround"] = int(co), int(cu), float(oround)
        out["crn_main_bookmaker_count"] = int(min(co, cu))
        if max_update is not None:
            out["crn_main_max_update_minutes"] = float(max_update)

    # —— 全线快照（用于“全线 Top 10”打印） —— #
    ou_all_list = []
    for line, sides in ou_map.items():
        ov = [x for x in sides.get("over", []) if x]
        un = [x for x in sides.get("under", []) if x]
        if not ov or not un:
            continue
        om, um = _median(ov), _median(un)
        if not om or not um:
            continue
        oround = 1.0/om + 1.0/um
        if 1.00 <= oround <= 1.25:
            bm_over = len(sides.get("_over_bm", []))
            bm_under = len(sides.get("_under_bm", []))
            co = bm_over if bm_over else len(ov)
            cu = bm_under if bm_under else len(un)
            updates = [
                *[x for x in sides.get("_over_updates", []) if x is not None],
                *[x for x in sides.get("_under_updates", []) if x is not None],
            ]
            max_update = float(max(updates)) if updates else None
            entry = {
                "line": float(line),
                "over": float(om),
                "under": float(um),
                "co": int(co),
                "cu": int(cu),
                "bookmakers": int(min(co, cu)),
                "overround": float(oround),
            }
            if max_update is not None:
                entry["max_update_minutes"] = max_update
            ou_all_list.append(entry)
    if ou_all_list:
        out["ou_all"] = sorted(ou_all_list, key=lambda x: x["line"])

    crn_all_list = []
    for line, sides in cr_map.items():
        ov = [x for x in sides.get("over", []) if x]
        un = [x for x in sides.get("under", []) if x]
        if not ov or not un:
            continue
        om, um = _median(ov), _median(un)
        if not om or not um:
            continue
        oround = 1.0/om + 1.0/um
        if 1.00 <= oround <= 1.30:
            bm_over = len(sides.get("_over_bm", []))
            bm_under = len(sides.get("_under_bm", []))
            co = bm_over if bm_over else len(ov)
            cu = bm_under if bm_under else len(un)
            updates = [
                *[x for x in sides.get("_over_updates", []) if x is not None],
                *[x for x in sides.get("_under_updates", []) if x is not None],
            ]
            max_update = float(max(updates)) if updates else None
            entry = {
                "line": float(line),
                "over": float(om),
                "under": float(um),
                "co": int(co),
                "cu": int(cu),
                "bookmakers": int(min(co, cu)),
                "overround": float(oround),
            }
            if max_update is not None:
                entry["max_update_minutes"] = max_update
            crn_all_list.append(entry)
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

    out["_raw_ou_map"]  = _clean_ou_or_cr_map(ou_map)
    out["_raw_crn_map"] = _clean_ou_or_cr_map(cr_map)
    out["_raw_ah_map"]  = _clean_ah_map(ah_map)

    # —— 已清洗“全线矩阵”摘要（中位数、样本、超额） —— #
    def _collect_updates(src_val: Any) -> List[float]:
        if isinstance(src_val, Sequence) and not isinstance(src_val, (str, bytes)):
            return [float(x) for x in src_val if x is not None]
        return []

    def _len_collection(src_val: Any) -> int:
        if isinstance(src_val, (set, list, tuple)):
            return len([x for x in src_val if x is not None])
        return 0

    def pack_ou_lines(src: Dict[float, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
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
            bm_over = _len_collection(sides.get("_over_bm"))
            bm_under = _len_collection(sides.get("_under_bm"))
            co = bm_over if bm_over else len(ov)
            cu = bm_under if bm_under else len(un)
            updates = _collect_updates(sides.get("_over_updates")) + _collect_updates(sides.get("_under_updates"))
            entry = {
                "over_median": float(om),
                "under_median": float(um),
                "over_cnt": int(co),
                "under_cnt": int(cu),
                "overround": float(overround),
                "bookmaker_count": int(min(co, cu)),
            }
            if updates:
                entry["max_update_minutes"] = float(max(updates))
            res[str(float(line))] = entry
        return res

    def pack_ah_lines(src: Dict[float, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
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
            bm_home = _len_collection(sides.get("_home_bm"))
            bm_away = _len_collection(sides.get("_away_bm"))
            ch = bm_home if bm_home else len(hs)
            ca = bm_away if bm_away else len(as_)
            updates = _collect_updates(sides.get("_home_updates")) + _collect_updates(sides.get("_away_updates"))
            entry = {
                "home_median": float(oh),
                "away_median": float(oa),
                "home_cnt": int(ch),
                "away_cnt": int(ca),
                "overround": float(overround),
                "bookmaker_count": int(min(ch, ca)),
            }
            if updates:
                entry["max_update_minutes"] = float(max(updates))
            res[str(float(line))] = entry
        return res

    out["ou_lines"]  = pack_ou_lines(ou_map)
    out["crn_lines"] = pack_ou_lines(cr_map)
    out["ah_lines"]  = pack_ah_lines(ah_map)

    return out

__all__ = [
    "fixtures_by_date", "list_teams_in_league", "team_statistics",
    "recent_fixtures_by_team", "find_league", "odds_by_fixture"
]
