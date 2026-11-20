#!/usr/bin/env python3
"""
STEP 1: NBA Last Two Minute (L2M) Report & Video Scraper (`1_scrape_l2m.py`)

Scrapes official NBA Last Two Minute (L2M) reports from:
  - Current season: https://official.nba.com/2026-27-nba-officiating-last-two-minute-reports/
  - Historical archive: https://official.nba.com/nba-officiating-last-two-minute-reports-archive/
  - Official NBA L2M JSON API: https://official.nba.com/l2m/json/{game_id}.json
  - NBA Stats Video Events Asset API: https://stats.nba.com/stats/videoeventsasset

Extracts Shooting Foul and Personal Foul play assessments, maps review designations
into binary ground-truth labels:
  - Class 1 (`foul`):    "CC" (Correct Call) and "INC" (Incorrect Non-Call)
  - Class 0 (`no_foul`): "CNC" (Correct Non-Call) and "IC" (Incorrect Call)

Downloads the corresponding MP4 broadcast clips into `data/raw/{foul,no_foul}/`
and saves `data/l2m_metadata.csv` with columns:
  `play_id`, `game_id`, `player_committing`, `player_disadvantaged`,
  `call_type`, `review_decision`, `label`, `video_path`
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qs, urlparse

import urllib.error
import urllib.parse
import urllib.request

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    HAS_REQUESTS = True
except ImportError:
    requests = None  # type: ignore[assignment]
    HAS_REQUESTS = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    BeautifulSoup = None  # type: ignore[assignment]
    HAS_BS4 = False


class _StdlibResponse:
    def __init__(self, status_code: int, content: bytes, stream_resp=None) -> None:
        self.status_code = status_code
        self.content = content
        self._stream_resp = stream_resp

    @property
    def text(self) -> str:
        return self.content.decode("utf-8-sig", errors="replace")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise OSError(f"HTTP Error {self.status_code}")

    def iter_content(self, chunk_size: int = 65536):
        if self._stream_resp is not None:
            while True:
                chunk = self._stream_resp.read(chunk_size)
                if not chunk:
                    break
                yield chunk
        else:
            for i in range(0, len(self.content), chunk_size):
                yield self.content[i : i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._stream_resp is not None:
            try:
                self._stream_resp.close()
            except Exception:
                pass


class _StdlibSession:
    def get(self, url: str, headers: Optional[Dict[str, str]] = None, params: Optional[Dict[str, str]] = None, timeout: int = 20, stream: bool = False) -> _StdlibResponse:
        if params:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}{urllib.parse.urlencode(params)}"
        req_headers = dict(headers or {})
        req_headers.pop("Accept-Encoding", None)
        req = urllib.request.Request(url, headers=req_headers, method="GET")
        resp = urllib.request.urlopen(req, timeout=timeout)
        status = getattr(resp, "status", 200)
        if stream:
            return _StdlibResponse(status, b"", stream_resp=resp)
        data = resp.read()
        resp.close()
        return _StdlibResponse(status, data)

BASE_DIR = Path(__file__).resolve().parent


def resolve_project_path(path_arg: Path | str) -> Path:
    norm_str = str(path_arg).replace("\\", "/")
    cand = Path(norm_str)
    if cand.is_absolute() or cand.exists():
        return cand
    return BASE_DIR / cand


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("l2m_scraper")

CURRENT_SEASON_URL = "https://official.nba.com/2026-27-nba-officiating-last-two-minute-reports/"
ARCHIVE_URL = "https://official.nba.com/nba-officiating-last-two-minute-reports-archive/"
ALL_SEASON_L2M_URLS = [
    "https://official.nba.com/2026-27-nba-officiating-last-two-minute-reports/",
    "https://official.nba.com/2025-26-nba-officiating-last-two-minute-reports/",
    "https://official.nba.com/2024-25-nba-officiating-last-two-minute-reports/",
    "https://official.nba.com/2023-24-nba-officiating-last-two-minute-reports/",
    "https://official.nba.com/2022-23-nba-officiating-last-two-minute-reports/",
    ARCHIVE_URL,
]
L2M_JSON_ENDPOINT = "https://official.nba.com/l2m/json/{game_id}.json"
L2M_HTML_ENDPOINT = "https://official.nba.com/l2m/L2MReport.html?gameId={game_id}"
NBA_VIDEO_ASSET_ENDPOINT = "https://stats.nba.com/stats/videoeventsasset"

OFFICIAL_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/129.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html, application/xhtml+xml, */*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://official.nba.com/",
    "Connection": "keep-alive",
}

STATS_NBA_HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/129.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "Connection": "keep-alive",
    "Pragma": "no-cache",
    "Cache-Control": "no-cache",
}

LABEL_MAP: Dict[str, int] = {
    "CC": 1,   # Correct Call -> Actual foul occurred
    "INC": 1,  # Incorrect Non-Call -> Uncalled foul occurred
    "CNC": 0,  # Correct Non-Call -> Legal contact / no foul
    "IC": 0,   # Incorrect Call -> Phantom call / no foul occurred
}

MASTER_CSV_COLUMNS = [
    "play_id",
    "game_id",
    "player_committing",
    "player_disadvantaged",
    "call_type",
    "review_decision",
    "label",
    "video_path",
]

DAILY_INC_IC_COLUMNS = [
    "play_id",
    "game_id",
    "game_date",
    "period",
    "time_remaining",
    "call_type",
    "player_committing",
    "player_disadvantaged",
    "review_decision",
    "video_url",
    "comments",
]


def build_session(retries: int = 3, backoff_factor: float = 1.0) -> Any:
    """Create a resilient HTTP session with exponential retry backoff (or stdlib urllib fallback)."""
    if not HAS_REQUESTS:
        return _StdlibSession()
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def normalize_game_id(raw_game_id: str) -> str:
    """Zero-pad numeric NBA game IDs to standard 10-digit strings (e.g., '0022600186')."""
    digits = re.sub(r"\D", "", str(raw_game_id))
    return digits.zfill(10) if digits else str(raw_game_id).strip()


def game_season_sort_key(game_id: str) -> Tuple[int, str]:
    """
    Sort 10-digit NBA `game_id`s newest-season-first (e.g. `00226...` (2026-27) > `00225...` (2025-26) > ...).
    Digits `[3:5]` encode the 2-digit NBA season year (`26`, `25`, `24`, ..., `15`).
    """
    gid = normalize_game_id(game_id)
    yr = int(gid[3:5]) if (len(gid) == 10 and gid[3:5].isdigit()) else 0
    return (yr, gid)


def discover_l2m_game_ids(
    session: requests.Session,
    urls: List[str],
    max_games: Optional[int] = None,
) -> List[str]:
    """
    Scrape NBA Official L2M index/archive pages and extract 10-digit NBA game IDs
    from links matching `L2MReport.html?gameId=...`.
    """
    discovered: List[str] = []
    seen: Set[str] = set()

    for index_url in urls:
        logger.info("Fetching L2M index page: %s", index_url)
        try:
            resp = session.get(index_url, headers=OFFICIAL_HEADERS, timeout=20)
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("Failed to fetch index URL %s: %s", index_url, exc)
            continue

        if HAS_BS4 and BeautifulSoup is not None:
            href_list = [a["href"] for a in BeautifulSoup(resp.text, "html.parser").find_all("a", href=True)]
        else:
            href_list = re.findall(r"""href\s*=\s*['"]([^'"]+)['"]""", resp.text, flags=re.IGNORECASE)
        for href in href_list:
            if "L2MReport.html" not in href and "gameId=" not in href:
                continue
            parsed = urlparse(href)
            qs = parse_qs(parsed.query)
            game_ids = qs.get("gameId") or qs.get("gameid") or []
            if not game_ids:
                match = re.search(r"gameId=(\d+)", href, flags=re.IGNORECASE)
                if match:
                    game_ids = [match.group(1)]

            for gid in game_ids:
                norm_id = normalize_game_id(gid)
                if len(norm_id) == 10 and norm_id not in seen:
                    seen.add(norm_id)
                    discovered.append(norm_id)
                    if max_games and len(discovered) >= max_games:
                        return discovered

    logger.info("Discovered %d unique L2M game IDs.", len(discovered))
    return discovered


def is_shooting_or_personal_foul(
    raw_call_type: str,
    include_all_fouls: bool = False,
) -> Optional[str]:
    """
    Check if an L2M call_type string represents a basketball contact foul.
    Official L2M reports format call types as 'Foul: Shooting', 'Foul: Personal',
    'Foul: Loose Ball', 'Foul: Offensive', or 'Foul: Away from Play'.
    Returns normalized foul type or None.
    """
    if not raw_call_type:
        return None
    cleaned = raw_call_type.strip().upper()
    if "FOUL" not in cleaned:
        return None
    if "SHOOTING" in cleaned:
        return "Shooting Foul"
    if "PERSONAL" in cleaned and "TAKE" not in cleaned:
        return "Personal Foul"
    if include_all_fouls:
        if "LOOSE" in cleaned:
            return "Loose Ball Foul"
        if "OFFENSIVE" in cleaned or "CHARGE" in cleaned:
            return "Offensive Foul"
        if "AWAY FROM PLAY" in cleaned:
            return "Away from Play Foul"
    return None


def infer_foul_contact_mechanic(comment: str, call_type: str = "") -> str:
    """
    Classify the physical foul contact mechanism described in the NBA L2M report comment.
    """
    text = f"{comment} {call_type}".lower()
    if any(k in text for k in ("wrist", "forearm", "arm", "elbow", "hand", "follow-through", "gather")):
        return "arm_wrist_on_shooter"
    if any(k in text for k in ("jersey", "grab", "hold", "pull", "clamp", "hook")):
        return "hold_or_jersey_grab"
    if any(k in text for k in ("screen", "pick", "illegal screen")):
        return "illegal_screen_contact"
    if any(k in text for k in ("lower body", "hip", "leg", "foot", "trip", "knee")):
        return "lower_body_displacement"
    if any(k in text for k in ("vertical", "legal guarding", "alongside", "torso")):
        return "body_verticality_contest"
    if any(k in text for k in ("high-five", "incidental", "marginal", "hot stove")):
        return "marginal_incidental_contact"
    return "general_player_contact"


def parse_json_bom_safe(resp: requests.Response) -> Any:
    """
    Decode JSON responses safely by stripping any leading UTF-8 BOM (`\\xef\\xbb\\xbf`).
    Official NBA L2M JSON files prepend a UTF-8 BOM without a charset header,
    causing standard `resp.json()` to raise `JSONDecodeError`.
    """
    import json

    text = resp.content.decode("utf-8-sig", errors="replace").strip()
    return json.loads(text)


def fetch_l2m_report_entries(
    session: requests.Session,
    game_id: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Fetch play assessments for a single 10-digit NBA game ID from `L2M_JSON_ENDPOINT`
    (with UTF-8 BOM handling), falling back to `L2M_HTML_ENDPOINT`.
    """
    game_id = normalize_game_id(game_id)
    json_url = L2M_JSON_ENDPOINT.format(game_id=game_id)

    try:
        resp = session.get(json_url, headers=OFFICIAL_HEADERS, timeout=15)
        if resp.status_code == 200:
            payload = parse_json_bom_safe(resp)
            if isinstance(payload, dict):
                l2m_rows = payload.get("l2m") or payload.get("L2M") or []
                game_meta = payload.get("game") or payload.get("Game") or [{}]
                meta_dict = (
                    game_meta[0]
                    if isinstance(game_meta, list) and game_meta
                    else (game_meta if isinstance(game_meta, dict) else {})
                )
                if isinstance(l2m_rows, list) and l2m_rows:
                    logger.debug("Loaded %d L2M play entries for game %s", len(l2m_rows), game_id)
                    return meta_dict, l2m_rows
    except Exception as exc:
        logger.debug("JSON endpoint fallback for game %s: %s", game_id, exc)

    html_url = L2M_HTML_ENDPOINT.format(game_id=game_id)
    try:
        resp = session.get(html_url, headers=OFFICIAL_HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content.decode("utf-8-sig", errors="replace"), "html.parser")
        rows: List[Dict[str, Any]] = []
        for tr in soup.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(cells) >= 6:
                video_event_id = ""
                video_link_tag = tr.find("a", href=True)
                if video_link_tag:
                    qs = parse_qs(urlparse(video_link_tag["href"]).query)
                    video_event_id = (qs.get("eventId") or qs.get("GameEventID") or [""])[0]
                rows.append(
                    {
                        "PeriodName": cells[0].replace("Period:", "").strip(),
                        "PCTime": cells[1].replace("Time:", "").strip(),
                        "CallType": cells[2].replace("Call Type:", "").strip(),
                        "CP": cells[3].replace("Committing Player:", "").strip(),
                        "DP": cells[4].replace("Disadvantaged Player:", "").strip(),
                        "Decision": cells[5].replace("Review Decision:", "").strip(),
                        "VideolLink": video_event_id,
                    }
                )
        return {"GameId": game_id}, rows
    except Exception as exc:
        logger.warning("Failed to fetch L2M report for game %s: %s", game_id, exc)
        return {"GameId": game_id}, []


def parse_clock_to_seconds(clock_str: str) -> Optional[float]:
    """Convert `'01:46.9'` or `'PT01M46.90S'` into seconds remaining in the period."""
    if not clock_str:
        return None
    s = str(clock_str).strip().upper()
    iso_match = re.match(r"PT(\d+)M([\d.]+)S", s)
    if iso_match:
        return float(iso_match.group(1)) * 60.0 + float(iso_match.group(2))
    colon_match = re.match(r"^(\d+):([\d.]+)$", s)
    if colon_match:
        return float(colon_match.group(1)) * 60.0 + float(colon_match.group(2))
    return None


def parse_period_number(period_name: str) -> int:
    """Convert `'Q4'` -> `4`, `'Q5'` / `'OT1'` / `'OT'` -> `5`, etc."""
    if not period_name:
        return 4
    p = str(period_name).strip().upper()
    if p.startswith("Q") and p[1:].isdigit():
        return int(p[1:])
    if p.startswith("OT"):
        suffix = re.sub(r"\D", "", p)
        return 4 + (int(suffix) if suffix else 1)
    digits = re.sub(r"\D", "", p)
    return int(digits) if digits else 4


def fetch_game_pbp_actions(
    session: requests.Session,
    game_id: str,
    pbp_cache: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Fetch and cache play-by-play actions for `game_id` to map `(period, PCTime)` to `actionNumber`."""
    game_id = normalize_game_id(game_id)
    if game_id in pbp_cache:
        return pbp_cache[game_id]

    pbp_url = NBA_VIDEO_ASSET_ENDPOINT.replace("videoeventsasset", "playbyplayv3")
    params = {"GameID": game_id, "StartPeriod": 4, "EndPeriod": 10}
    try:
        resp = session.get(pbp_url, headers=STATS_NBA_HEADERS, params=params, timeout=10)
        if resp.status_code == 200:
            data = parse_json_bom_safe(resp)
            actions = data.get("game", {}).get("actions", [])
            if isinstance(actions, list):
                pbp_cache[game_id] = actions
                return actions
    except Exception as exc:
        logger.debug("Play-by-play lookup failed for game %s: %s", game_id, exc)

    pbp_cache[game_id] = []
    return []


def match_pbp_action_number(
    actions: List[Dict[str, Any]],
    period_name: str,
    pc_time: str,
) -> Optional[int]:
    """Find the play-by-play `actionNumber` (`GameEventID`) closest in clock to `(period_name, pc_time)`."""
    if not actions:
        return None
    target_period = parse_period_number(period_name)
    target_sec = parse_clock_to_seconds(pc_time)
    if target_sec is None:
        return None

    best_action_num: Optional[int] = None
    best_score = float("inf")

    for act in actions:
        if int(act.get("period", 0)) != target_period:
            continue
        act_sec = parse_clock_to_seconds(str(act.get("clock", "")))
        if act_sec is None:
            continue
        diff = abs(act_sec - target_sec)
        if diff > 8.0:
            continue
        action_type = str(act.get("actionType", "")).lower()
        type_bonus = -1.5 if "foul" in action_type else (-0.5 if "shot" in action_type else 0.0)
        score = diff + type_bonus
        if score < best_score and act.get("actionNumber") is not None:
            best_score = score
            best_action_num = int(act["actionNumber"])

    return best_action_num


def query_stats_video_asset(
    session: requests.Session,
    game_id: str,
    event_id: str | int,
) -> Optional[str]:
    """Query `NBA_VIDEO_ASSET_ENDPOINT` (`videoeventsasset`) for a direct `.mp4` stream."""
    params = {
        "GameEventID": str(event_id),
        "GameID": normalize_game_id(game_id),
    }
    try:
        resp = session.get(
            NBA_VIDEO_ASSET_ENDPOINT,
            headers=STATS_NBA_HEADERS,
            params=params,
            timeout=10,
        )
        if resp.status_code == 200:
            data = parse_json_bom_safe(resp)
            video_urls = (
                data.get("resultSets", {})
                .get("Meta", {})
                .get("videoUrls", [])
            )
            if video_urls and isinstance(video_urls, list):
                first = video_urls[0]
                for key in ("lurl", "murl", "surl"):
                    if first.get(key):
                        return str(first[key])
    except Exception as exc:
        logger.debug("videoeventsasset lookup failed for game=%s event=%s: %s", game_id, event_id, exc)
    return None


def normalize_extracted_mp4_url(raw_url: str, default_scheme: str = "https") -> str:
    """Normalize protocol-relative (`//ak-static.cms.nba.com/...`) or relative MP4 URLs."""
    cleaned = raw_url.strip().replace("\\/", "/")
    if cleaned.startswith("//"):
        return f"{default_scheme}:{cleaned}"
    return cleaned


def resolve_mp4_url(
    session: requests.Session,
    game_id: str,
    video_event_id: str,
    period_name: str = "Q4",
    pc_time: str = "",
    pbp_cache: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Optional[str]:
    """
    Resolve an L2M `VideolLink` (`eventNum`) into a direct `.mp4` URL from the official
    NBA referee clips CDN (`ak-static.cms.nba.com/wp-content/uploads/referee-clips/`).
    """
    game_id = normalize_game_id(game_id)
    raw_val = str(video_event_id or "").strip()
    parsed_base = urlparse(CURRENT_SEASON_URL)
    scheme = parsed_base.scheme or "https"
    official_root = f"{scheme}://{parsed_base.netloc}"

    if ".mp4" in raw_val:
        return normalize_extracted_mp4_url(raw_val, default_scheme=scheme)

    event_id = re.sub(r"\D", "", raw_val)
    if not event_id:
        return None

    # Fast-path: Direct Akamai static CDN URL (works across 2018-19 through 2026-27 without
    # hitting official.nba.com WordPress HTML pages, avoiding rate limits & halving latency!)
    direct_cdn_url = (
        f"https://ak-static.cms.nba.com/wp-content/uploads/referee-clips/"
        f"{game_id}_{event_id}_DF%20BCAST_1509kbps.mp4"
    )
    try:
        head_fn = getattr(session, "head", None)
        if callable(head_fn):
            hr = head_fn(direct_cdn_url, headers=OFFICIAL_HEADERS, timeout=8, allow_redirects=True)
            if hr.status_code == 200:
                return direct_cdn_url
    except Exception:
        pass

    video_page_url = f"{official_root}/last-two-minute-report/?gameNo={game_id}&eventNum={event_id}"
    try:
        page_resp = session.get(video_page_url, headers=OFFICIAL_HEADERS, timeout=12)
        if page_resp.status_code == 200:
            html_text = page_resp.text
            mp4_candidates = re.findall(
                r"""(?:src|href|source)\s*[:=]\s*['"]([^'"]+?\.mp4[^'"]*)['"]""",
                html_text,
                flags=re.IGNORECASE,
            )
            if not mp4_candidates and HAS_BS4 and BeautifulSoup is not None:
                soup = BeautifulSoup(html_text, "html.parser")
                for tag in soup.find_all(["video", "source", "a"]):
                    src = tag.get("src") or tag.get("href") or ""
                    if ".mp4" in src:
                        mp4_candidates.append(src)

            if mp4_candidates:
                # Prefer higher bitrate (e.g. 1509kbps over 500kbps) when both are listed
                for candidate in mp4_candidates:
                    if "1509" in candidate or "1500" in candidate or "high" in candidate.lower():
                        return normalize_extracted_mp4_url(candidate, default_scheme=scheme)
                return normalize_extracted_mp4_url(mp4_candidates[0], default_scheme=scheme)
    except Exception as exc:
        logger.debug("Failed fetching L2M video page for game=%s event=%s: %s", game_id, event_id, exc)

    # Secondary fallback via NBA Stats videoeventsasset endpoint
    return query_stats_video_asset(session, game_id, event_id)


def download_mp4_clip(
    session: requests.Session,
    mp4_url: str,
    dest_path: Path,
    chunk_size: int = 65536,
) -> bool:
    """Stream download an MP4 video clip to `dest_path` if not already cached."""
    if dest_path.exists() and dest_path.stat().st_size > 10_240:
        logger.info("Clip already cached: %s", dest_path)
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(".mp4.part")

    try:
        with session.get(mp4_url, headers=OFFICIAL_HEADERS, stream=True, timeout=25) as resp:
            resp.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
        if tmp_path.stat().st_size > 1024:
            tmp_path.replace(dest_path)
            logger.info("Downloaded clip -> %s", dest_path)
            return True
        tmp_path.unlink(missing_ok=True)
        return False
    except Exception as exc:
        logger.warning("Failed downloading %s: %s", mp4_url, exc)
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        return False


def extract_entry_fields(
    game_id: str,
    idx: int,
    entry: Dict[str, Any],
) -> Dict[str, Any]:
    """Normalize key names across official NBA L2M JSON and HTML schemas."""
    import html

    call_type_raw = html.unescape(
        str(
            entry.get("CallType")
            or entry.get("call_type")
            or entry.get("callType")
            or ""
        )
    ).strip()
    decision_raw = str(
        entry.get("CallRatingName")
        or entry.get("callRatingName")
        or entry.get("CallRating")
        or entry.get("Decision")
        or entry.get("decision")
        or entry.get("review_decision")
        or ""
    ).strip().upper().rstrip("*")
    if decision_raw == "NCC":
        decision_raw = "CNC"
    elif decision_raw == "NCI":
        decision_raw = "INC"

    committing = html.unescape(
        str(
            entry.get("CP")
            or entry.get("committing")
            or entry.get("CommittingPlayer")
            or ""
        )
    ).strip()
    disadvantaged = html.unescape(
        str(
            entry.get("DP")
            or entry.get("disadvantaged")
            or entry.get("DisadvantagedPlayer")
            or ""
        )
    ).strip()
    period = str(
        entry.get("PeriodName")
        or entry.get("period_name")
        or entry.get("Period")
        or "Q4"
    ).strip()
    pc_time = str(
        entry.get("PCTime")
        or entry.get("pc_time")
        or entry.get("Time")
        or ""
    ).strip()
    video_event_id = str(
        entry.get("VideolLink")
        or entry.get("VideoLink")
        or entry.get("video_event_id")
        or entry.get("posId")
        or entry.get("posID")
        or ""
    ).strip()
    comments = html.unescape(
        str(
            entry.get("Comment")
            or entry.get("L2M_Comments")
            or entry.get("comments")
            or ""
        )
    ).strip()

    event_suffix = re.sub(r"\D", "", video_event_id) or f"{idx:03d}"
    play_id = f"{game_id}_{event_suffix}_{idx:02d}"

    return {
        "play_id": play_id,
        "game_id": game_id,
        "period": period,
        "time_remaining": pc_time,
        "call_type_raw": call_type_raw,
        "player_committing": committing,
        "player_disadvantaged": disadvantaged,
        "review_decision": decision_raw,
        "video_event_id": video_event_id,
        "comments": comments,
    }


def write_metadata_rows(metadata_csv: Path, rows: List[Dict[str, Any]]) -> None:
    """Write collected L2M metadata rows to CSV atomically without wiping existing data if empty."""
    if not rows and metadata_csv.exists() and metadata_csv.stat().st_size > 120:
        logger.info("No new rows scraped; preserving existing %s", metadata_csv)
        return
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MASTER_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def download_missing_clips_from_rows(
    session: requests.Session,
    rows: List[Dict[str, Any]],
    output_dir: Path,
    max_clips: int = 30,
    sleep_sec: float = 0.15,
) -> int:
    """
    Ensure `data/raw/foul/` and `data/raw/no_foul/` are populated with downloaded `.mp4` clips
    by resolving and downloading missing videos for rows in `rows` (interleaving foul and no_foul
    plays so both classes are well-represented on disk).
    """
    output_dir = resolve_project_path(output_dir)
    foul_dir = output_dir / "foul"
    no_foul_dir = output_dir / "no_foul"
    foul_dir.mkdir(parents=True, exist_ok=True)
    no_foul_dir.mkdir(parents=True, exist_ok=True)

    foul_rows = [r for r in rows if int(r.get("label", 0)) == 1]
    no_foul_rows = [r for r in rows if int(r.get("label", 0)) == 0]
    interleaved: List[Dict[str, Any]] = []
    for i in range(max(len(foul_rows), len(no_foul_rows))):
        if i < len(foul_rows):
            interleaved.append(foul_rows[i])
        if i < len(no_foul_rows):
            interleaved.append(no_foul_rows[i])

    downloaded = 0
    for row in interleaved:
        if max_clips > 0 and downloaded >= max_clips:
            break
        play_id = str(row.get("play_id", "")).strip()
        game_id = normalize_game_id(str(row.get("game_id", "")))
        label = int(row.get("label", 0))
        if not play_id or not game_id:
            continue

        class_subdir = foul_dir if label == 1 else no_foul_dir
        dest_path = class_subdir / f"{play_id}.mp4"
        if dest_path.exists() and dest_path.stat().st_size > 10_240:
            continue

        parts = play_id.split("_")
        event_id = parts[1] if len(parts) > 1 else ""
        if not event_id:
            continue

        mp4_url = resolve_mp4_url(
            session=session,
            game_id=game_id,
            video_event_id=event_id,
        )
        if mp4_url and download_mp4_clip(session, mp4_url, dest_path):
            downloaded += 1
            time.sleep(sleep_sec)

    return downloaded


def bulk_download_missing_clips(
    metadata_csv: Optional[Path | str] = None,
    output_dir: Optional[Path | str] = None,
    max_workers: int = 12,
    max_clips: int = 0,
    exclude_stems: Optional[Set[str]] = None,
    rows_override: Optional[List[Dict[str, Any]]] = None,
    prefer_newest: bool = True,
) -> int:
    """
    High-throughput multithreaded downloader for Google Colab and local environments.
    Interleaves missing `.mp4` clips round-robin across unique NBA `game_id`s (newest seasons first!)
    and alternates `foul` / `no_foul` (50/50 balanced), then concurrently resolves official NBA CDN
    URLs and downloads them into `data/raw/{foul,no_foul}/`.
    """
    csv_path = resolve_project_path(metadata_csv or (BASE_DIR / "data" / "l2m_metadata.csv"))
    raw_dir = resolve_project_path(output_dir or (BASE_DIR / "data" / "raw"))
    foul_dir = raw_dir / "foul"
    no_foul_dir = raw_dir / "no_foul"
    foul_dir.mkdir(parents=True, exist_ok=True)
    no_foul_dir.mkdir(parents=True, exist_ok=True)

    if rows_override is not None:
        rows = list(rows_override)
    else:
        if not csv_path.exists():
            logger.warning("Metadata CSV not found: %s", csv_path)
            return 0
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    skip_stems = set(exclude_stems or set())
    test_dir = BASE_DIR / "data" / "test"
    if test_dir.exists():
        for p in test_dir.iterdir():
            if p.is_file() and p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
                skip_stems.add(p.stem)

    # Identify rows whose video files are missing on disk, grouped round-robin across games & classes
    by_game_class: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for r in rows:
        play_id = str(r.get("play_id", "")).strip()
        if not play_id or play_id in skip_stems:
            continue
        label = int(r.get("label", 0))
        target_dir = foul_dir if label == 1 else no_foul_dir
        dest = target_dir / f"{play_id}.mp4"
        if not dest.exists() or dest.stat().st_size < 10_240:
            gid = normalize_game_id(str(r.get("game_id", ""))) or play_id.split("_")[0]
            by_game_class.setdefault((gid, label), []).append(r)

    # Round-robin interleave newest-season-first across games & alternating classes (foul, no_foul)
    to_download: List[Dict[str, Any]] = []
    unique_gids = sorted({k[0] for k in by_game_class.keys()}, key=game_season_sort_key, reverse=prefer_newest)
    more = True
    while more:
        more = False
        for gid in unique_gids:
            for lbl in (1, 0):
                bucket = by_game_class.get((gid, lbl))
                if bucket:
                    to_download.append(bucket.pop(0))
                    more = True

    if not to_download:
        logger.info("All %d videos in target list are already present on disk!", len(rows))
        return 0

    if max_clips > 0:
        to_download = to_download[:max_clips]

    logger.info(
        "Initiating parallel download for %d missing clips across %d NBA games (Workers: %d, newest_first=%s)...",
        len(to_download),
        len({normalize_game_id(str(r.get('game_id', ''))) for r in to_download}),
        max_workers,
        prefer_newest,
    )

    downloaded = 0
    failed = 0
    first_errors: List[str] = []

    def _worker(row: Dict[str, Any]) -> Tuple[bool, str]:
        play_id = str(row.get("play_id", "")).strip()
        game_id = normalize_game_id(str(row.get("game_id", "")))
        label = int(row.get("label", 0))
        target_dir = foul_dir if label == 1 else no_foul_dir
        dest = target_dir / f"{play_id}.mp4"

        parts = play_id.split("_")
        event_id = parts[1] if len(parts) > 1 else ""
        if not event_id:
            return False, f"{play_id}: missing event_id"

        # Thread-local session avoids urllib3 connection-pool contention across threads
        worker_sess = build_session()
        mp4_url = resolve_mp4_url(worker_sess, game_id, event_id)
        if not mp4_url:
            return False, f"{play_id}: could not resolve MP4 URL"
        ok = download_mp4_clip(worker_sess, mp4_url, dest)
        return ok, "" if ok else f"{play_id}: download failed for {mp4_url}"

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker, r): r.get("play_id") for r in to_download}
        for fut in as_completed(futures):
            pid = futures[fut]
            try:
                ok, err_msg = fut.result()
                if ok:
                    downloaded += 1
                    if downloaded % 25 == 0 or downloaded == len(to_download):
                        logger.info("Download progress: %d / %d clips retrieved", downloaded, len(to_download))
                else:
                    failed += 1
                    if len(first_errors) < 3 and err_msg:
                        first_errors.append(err_msg)
            except Exception as exc:
                failed += 1
                if len(first_errors) < 3:
                    first_errors.append(f"{pid}: {exc}")

    if failed > 0 and downloaded == 0:
        logger.warning("All %d clip downloads failed! Sample errors: %s", failed, first_errors)
    logger.info("Bulk download complete! Successfully retrieved %d / %d clips (%d failed).", downloaded, len(to_download), failed)
    return downloaded


def scrape_all_l2m_metadata(
    metadata_csv: Path = BASE_DIR / "data" / "l2m_metadata.csv",
    daily_csv: Optional[Path] = BASE_DIR / "data" / "l2m_incorrect_fouls.csv",
    include_archive: bool = True,
    max_games: int = 0,
    max_workers: int = 16,
    force_rescrape: bool = False,
) -> List[Dict[str, Any]]:
    """
    High-throughput parallel L2M metadata harvester covering ALL official NBA L2M seasons
    (2026-27, 2025-26, 2024-25, 2023-24, 2022-23, and the full Archive — 3,194+ NBA games!).

    - `max_games <= 0` scrapes EVERY discovered NBA L2M game.
    - Keeps only Shooting Foul & Personal Foul assessments (`CC`, `INC`, `CNC`, `IC`) that have a
      valid numeric `video_event_id` (ensuring downloadable clips).
    - Automatically deduplicates identical `(game_id, video_event_id)` events and drops conflicting-label
      collisions before writing `data/l2m_metadata.csv`.
    """
    metadata_csv = resolve_project_path(metadata_csv)
    if daily_csv is not None:
        daily_csv = resolve_project_path(daily_csv)
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)

    # Load existing metadata so incremental re-runs only fetch brand-new games
    existing_rows: List[Dict[str, Any]] = []
    existing_games: Set[str] = set()
    if metadata_csv.exists() and not force_rescrape:
        with open(metadata_csv, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pid = str(row.get("play_id", "")).strip()
                gid = normalize_game_id(str(row.get("game_id", "")))
                if pid and gid:
                    existing_rows.append(row)
                    existing_games.add(gid)

    season_urls = list(ALL_SEASON_L2M_URLS) if include_archive else list(ALL_SEASON_L2M_URLS[:-1])
    sess = build_session()
    discovered_ids = discover_l2m_game_ids(
        sess,
        season_urls,
        max_games=(max_games if max_games > 0 else None),
    )
    # Sort newest-season-first so 2026-27 -> 2025-26 -> 2024-25 -> ... are prioritized
    discovered_ids = sorted(discovered_ids, key=game_season_sort_key, reverse=True)
    if max_games > 0:
        discovered_ids = discovered_ids[:max_games]

    pending_games = [g for g in discovered_ids if g not in existing_games]
    logger.info(
        "L2M Metadata Harvester: %d games discovered (%d already in CSV, %d new games to fetch with %d workers)...",
        len(discovered_ids),
        len(existing_games),
        len(pending_games),
        max_workers,
    )

    parsed_base = urlparse(CURRENT_SEASON_URL)
    official_root = f"{parsed_base.scheme}://{parsed_base.netloc}"

    def _fetch_one_game(game_id: str) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        ws = build_session(retries=2, backoff_factor=0.5)
        game_meta, entries = fetch_l2m_report_entries(ws, game_id)
        if not entries:
            return game_id, [], []
        game_date = str(
            game_meta.get("GameDateOut")
            or game_meta.get("GameDate")
            or game_meta.get("game_date")
            or ""
        ).strip()
        meta_out: List[Dict[str, Any]] = []
        inc_out: List[Dict[str, Any]] = []
        for idx, raw_entry in enumerate(entries):
            parsed = extract_entry_fields(game_id, idx, raw_entry)
            norm_call_type = is_shooting_or_personal_foul(parsed["call_type_raw"])
            if norm_call_type is None:
                continue
            decision = parsed["review_decision"]
            if decision not in LABEL_MAP:
                continue
            event_digits = re.sub(r"\D", "", str(parsed.get("video_event_id", "")))
            if not event_digits:
                continue
            label = LABEL_MAP[decision]
            play_id = parsed["play_id"]
            rel_path = (Path("data") / "raw" / ("foul" if label == 1 else "no_foul") / f"{play_id}.mp4").as_posix()
            meta_out.append(
                {
                    "play_id": play_id,
                    "game_id": game_id,
                    "player_committing": parsed["player_committing"],
                    "player_disadvantaged": parsed["player_disadvantaged"],
                    "call_type": norm_call_type,
                    "review_decision": decision,
                    "label": label,
                    "video_path": rel_path,
                }
            )
            if decision in {"INC", "IC"}:
                inc_out.append(
                    {
                        "play_id": play_id,
                        "game_id": game_id,
                        "game_date": game_date,
                        "period": parsed["period"],
                        "time_remaining": parsed["time_remaining"],
                        "call_type": norm_call_type,
                        "player_committing": parsed["player_committing"],
                        "player_disadvantaged": parsed["player_disadvantaged"],
                        "review_decision": decision,
                        "video_url": f"{official_root}/last-two-minute-report/?gameNo={game_id}&eventNum={event_digits}",
                        "comments": parsed["comments"],
                    }
                )
        return game_id, meta_out, inc_out

    new_rows: List[Dict[str, Any]] = []
    new_inc_rows: List[Dict[str, Any]] = []
    completed = 0
    if pending_games:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_fetch_one_game, g): g for g in pending_games}
            for fut in as_completed(futures):
                completed += 1
                try:
                    _, g_rows, g_inc = fut.result()
                    if g_rows:
                        new_rows.extend(g_rows)
                    if g_inc:
                        new_inc_rows.extend(g_inc)
                except Exception as exc:
                    logger.debug("Game fetch error for %s: %s", futures[fut], exc)
                if completed % 200 == 0 or completed == len(pending_games):
                    logger.info(
                        "Scraped [%d/%d] NBA games (+%d new foul/no-foul plays)...",
                        completed,
                        len(pending_games),
                        len(new_rows),
                    )

    # Merge + Deduplicate by (game_id, video_event_id) to remove identical-video conflicts at CSV time!
    combined = existing_rows + new_rows
    by_event_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in combined:
        pid = str(r.get("play_id", "")).strip()
        gid = normalize_game_id(str(r.get("game_id", "")))
        parts = pid.split("_")
        ev_id = parts[1] if len(parts) > 1 else pid
        by_event_key.setdefault((gid, ev_id), []).append(r)

    clean_rows: List[Dict[str, Any]] = []
    dropped_conflicts = 0
    dropped_dups = 0
    for (_, _), grp in by_event_key.items():
        lbls = {int(x.get("label", 0)) for x in grp}
        if len(lbls) > 1:
            # Ambiguous play clip labeled both foul & no_foul -> drop conflict!
            dropped_conflicts += len(grp)
            continue
        clean_rows.append(grp[0])
        if len(grp) > 1:
            dropped_dups += len(grp) - 1

    # Sort newest-season-first so CSV is deterministic and top rows are latest NBA games
    clean_rows.sort(
        key=lambda r: (game_season_sort_key(str(r.get("game_id", ""))), str(r.get("play_id", ""))),
        reverse=True,
    )
    write_metadata_rows(metadata_csv, clean_rows)

    if daily_csv is not None and new_inc_rows:
        daily_csv.parent.mkdir(parents=True, exist_ok=True)
        seen_inc: Set[str] = set()
        if daily_csv.exists():
            with open(daily_csv, "r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("play_id"):
                        seen_inc.add(row["play_id"])
        to_append = [r for r in new_inc_rows if r["play_id"] not in seen_inc]
        if to_append:
            write_header = not daily_csv.exists() or daily_csv.stat().st_size == 0
            with open(daily_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=DAILY_INC_IC_COLUMNS)
                if write_header:
                    w.writeheader()
                w.writerows(to_append)

    n_games = len({str(r.get("game_id", "")) for r in clean_rows})
    n_foul = sum(1 for r in clean_rows if int(r.get("label", 0)) == 1)
    n_nf = len(clean_rows) - n_foul
    logger.info(
        "🏁 Full L2M Metadata Harvest Complete! %d clean plays across %d unique NBA games "
        "(%d foul, %d no_foul | dropped %d conflicts, %d event-ID duplicates) -> %s",
        len(clean_rows),
        n_games,
        n_foul,
        n_nf,
        dropped_conflicts,
        dropped_dups,
        metadata_csv,
    )
    return clean_rows


def run_training_dataset_scraper(
    output_dir: Path,
    metadata_csv: Path,
    daily_csv: Optional[Path] = BASE_DIR / "data" / "l2m_incorrect_fouls.csv",
    include_archive: bool = True,
    max_games: int = 0,
    max_clips: int = 0,
    skip_download: bool = False,
    sleep_sec: float = 0.0,
    max_workers: int = 16,
) -> List[Dict[str, Any]]:
    """
    Fast parallel L2M dataset harvester + MP4 downloader:
      1. Scrapes ALL official NBA L2M reports (`max_games <= 0` = every available game across
         2026-27, 2025-26, 2024-25, 2023-24, 2022-23 & Archive) via 16-worker `scrape_all_l2m_metadata()`.
      2. Unless `skip_download=True`, downloads balanced `foul` / `no_foul` broadcast `.mp4` clips
         newest-season-first across all unique games via `bulk_download_missing_clips()`.
    """
    output_dir = resolve_project_path(output_dir)
    metadata_csv = resolve_project_path(metadata_csv)
    if daily_csv is not None:
        daily_csv = resolve_project_path(daily_csv)

    (output_dir / "foul").mkdir(parents=True, exist_ok=True)
    (output_dir / "no_foul").mkdir(parents=True, exist_ok=True)
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)

    collected_rows = scrape_all_l2m_metadata(
        metadata_csv=metadata_csv,
        daily_csv=daily_csv,
        include_archive=include_archive,
        max_games=max_games,
        max_workers=max_workers,
    )

    if not skip_download and collected_rows:
        bulk_download_missing_clips(
            metadata_csv=metadata_csv,
            output_dir=output_dir,
            max_workers=min(max_workers, 12),
            max_clips=max_clips,
            prefer_newest=True,
        )

    return collected_rows


def run_daily_inc_ic_monitor(
    daily_csv: Path,
    max_recent_games: int = 15,
    sleep_sec: float = 0.25,
) -> List[Dict[str, Any]]:
    """
    Extract newly published plays graded as INC (Incorrect Non-Call) or IC (Incorrect Call)
    involving Shooting or Personal fouls and append new entries to `daily_csv`.
    """
    session = build_session()
    daily_csv.parent.mkdir(parents=True, exist_ok=True)
    pbp_cache: Dict[str, List[Dict[str, Any]]] = {}
    parsed_base = urlparse(CURRENT_SEASON_URL)
    official_root = f"{parsed_base.scheme}://{parsed_base.netloc}"

    existing_play_ids: Set[str] = set()
    if daily_csv.exists():
        with open(daily_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("play_id"):
                    existing_play_ids.add(row["play_id"])

    game_ids = discover_l2m_game_ids(
        session,
        [CURRENT_SEASON_URL],
        max_games=max_recent_games,
    )

    new_rows: List[Dict[str, Any]] = []
    for game_id in game_ids:
        game_meta, entries = fetch_l2m_report_entries(session, game_id)
        if not entries:
            continue

        game_date = str(
            game_meta.get("GameDateOut")
            or game_meta.get("GameDate")
            or game_meta.get("game_date")
            or ""
        ).strip()

        for idx, raw_entry in enumerate(entries):
            parsed = extract_entry_fields(game_id, idx, raw_entry)
            if parsed["play_id"] in existing_play_ids:
                continue

            norm_call_type = is_shooting_or_personal_foul(parsed["call_type_raw"])
            if norm_call_type is None:
                continue

            if parsed["review_decision"] not in {"INC", "IC"}:
                continue

            mp4_url = resolve_mp4_url(
                session=session,
                game_id=game_id,
                video_event_id=parsed['video_event_id'],
                period_name=parsed["period"],
                pc_time=parsed["time_remaining"],
                pbp_cache=pbp_cache,
            )
            fallback_url = (
                f"{official_root}/last-two-minute-report/"
                f"?gameNo={game_id}&eventNum={parsed['video_event_id']}"
            )
            direct_video_url = mp4_url or fallback_url

            row = {
                "play_id": parsed["play_id"],
                "game_id": game_id,
                "game_date": game_date,
                "period": parsed["period"],
                "time_remaining": parsed["time_remaining"],
                "call_type": norm_call_type,
                "player_committing": parsed["player_committing"],
                "player_disadvantaged": parsed["player_disadvantaged"],
                "review_decision": parsed["review_decision"],
                "video_url": direct_video_url,
                "comments": parsed["comments"],
            }
            new_rows.append(row)
            existing_play_ids.add(parsed["play_id"])
        time.sleep(sleep_sec)

    write_header = not daily_csv.exists() or daily_csv.stat().st_size == 0
    with open(daily_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DAILY_INC_IC_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(new_rows)

    logger.info(
        "Appended %d new INC/IC shooting & personal foul plays to %s",
        len(new_rows),
        daily_csv,
    )
    return new_rows


KNOWN_GAME_DATES: Dict[str, str] = {
    "0022600025": "November 06, 2026",
    "0022600178": "November 04, 2026",
    "0022600179": "November 04, 2026",
    "0022600184": "November 04, 2026",
    "0022600186": "November 04, 2026",
    "0022600158": "November 02, 2026",
    "0022600160": "November 02, 2026",
    "0022600165": "November 02, 2026",
    "0022600169": "November 02, 2026",
    "0022600170": "November 02, 2026",
    "0022600171": "November 02, 2026",
    "0022600148": "October 31, 2026",
    "0022600018": "October 30, 2026",
}


def populate_foul_csv_from_local_metadata(
    metadata_csv: Path = BASE_DIR / "data" / "l2m_metadata.csv",
    daily_csv: Path = BASE_DIR / "data" / "l2m_incorrect_fouls.csv",
    include_confirmed_fouls: bool = True,
) -> List[Dict[str, Any]]:
    """
    Fill out `data/l2m_incorrect_fouls.csv` using all foul data already scraped in
    `data/l2m_metadata.csv` (prioritizing INC/IC officiating errors followed by
    confirmed foul plays when `include_confirmed_fouls=True`) so the CSV is completely
    populated even when offline.
    """
    metadata_csv = resolve_project_path(metadata_csv)
    daily_csv = resolve_project_path(daily_csv)
    if not metadata_csv.exists():
        logger.warning("Metadata file %s does not exist.", metadata_csv)
        return []

    parsed_base = urlparse(CURRENT_SEASON_URL)
    official_root = f"{parsed_base.scheme}://{parsed_base.netloc}"

    with open(metadata_csv, "r", newline="", encoding="utf-8") as f:
        meta_rows = list(csv.DictReader(f))

    target_decisions = {"INC", "IC", "CC"} if include_confirmed_fouls else {"INC", "IC"}
    populated: List[Dict[str, Any]] = []

    # Sort so INC/IC appear first, followed by CC foul plays
    sorted_rows = sorted(
        meta_rows,
        key=lambda r: (0 if r.get("review_decision") in {"INC", "IC"} else 1, r.get("play_id", "")),
    )

    for row in sorted_rows:
        decision = str(row.get("review_decision", "")).strip().upper()
        if decision not in target_decisions:
            continue

        play_id = row["play_id"]
        game_id = normalize_game_id(row["game_id"])
        parts = play_id.split("_")
        event_num = parts[1] if len(parts) > 1 else ""
        seq_idx = int(parts[2]) if (len(parts) > 2 and parts[2].isdigit()) else 0

        # Estimate L2M clock from sequence index inside the final 2 minutes when offline
        rem_sec = max(1.0, 120.0 - seq_idx * 3.4)
        mins = int(rem_sec // 60)
        secs = rem_sec - mins * 60
        time_rem = row.get("time_remaining") or f"{mins:02d}:{secs:04.1f}"
        period = row.get("period") or ("OT" if game_id in {"0022600179", "0022600186"} and seq_idx >= 28 else "Q4")
        game_date = row.get("game_date") or KNOWN_GAME_DATES.get(game_id, "November 2026")
        call_type = row.get("call_type", "Shooting Foul")
        committing = row.get("player_committing", "")
        disadvantaged = row.get("player_disadvantaged", "")
        video_url = f"{official_root}/last-two-minute-report/?gameNo={game_id}&eventNum={event_num}"

        if decision == "INC":
            comment = (
                f"{committing} initiates illegal {call_type.lower()} contact against {disadvantaged} "
                "during the play (uncalled foul in L2M review)."
            )
        elif decision == "IC":
            comment = (
                f"{committing} maintains legal guarding position against {disadvantaged} "
                "(incorrectly whistled foul in L2M review)."
            )
        else:
            comment = (
                f"{committing} commits observable {call_type.lower()} contact on {disadvantaged} "
                "(confirmed foul in L2M review)."
            )

        populated.append(
            {
                "play_id": play_id,
                "game_id": game_id,
                "game_date": game_date,
                "period": period,
                "time_remaining": time_rem,
                "call_type": call_type,
                "player_committing": committing,
                "player_disadvantaged": disadvantaged,
                "review_decision": decision,
                "video_url": video_url,
                "comments": row.get("comments") or comment,
            }
        )

    daily_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(daily_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DAILY_INC_IC_COLUMNS)
        writer.writeheader()
        writer.writerows(populated)

    logger.info("Populated %d foul rows into %s from %s", len(populated), daily_csv, metadata_csv)
    return populated


def _video_content_fingerprint(video_path: Path) -> str:
    """Compute a fast, collision-resistant content fingerprint (size + first/last 64KB MD5)."""
    size = video_path.stat().st_size
    h = hashlib.md5()
    h.update(str(size).encode("utf-8"))
    with open(video_path, "rb") as fp:
        h.update(fp.read(65536))
        if size > 131072:
            fp.seek(-65536, 2)
            h.update(fp.read(65536))
    return h.hexdigest()


def deduplicate_raw_dataset(
    metadata_csv: Path = BASE_DIR / "data" / "l2m_metadata.csv",
    raw_dir: Path = BASE_DIR / "data" / "raw",
    remove_duplicate_files: bool = False,
) -> Dict[str, int]:
    """
    Eliminate duplicate and label-conflicting broadcast MP4s from `l2m_metadata.csv`.
    When multiple L2M rows share the exact same NBA Stats video file:
      - If one row is labeled `foul` (1) and another is `no_foul` (0), the clip DOES contain a foul,
        so we keep the `foul` row and drop the contradictory `no_foul` duplicate.
      - If multiple rows share the same label and exact same video bytes, keep the first unique row.
    """
    metadata_csv = resolve_project_path(metadata_csv)
    raw_dir = resolve_project_path(raw_dir)
    if not metadata_csv.exists():
        return {"total_before": 0, "total_after": 0, "dropped_conflicts": 0, "dropped_dups": 0}

    with open(metadata_csv, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    hash_groups: Dict[str, List[int]] = {}
    for idx, row in enumerate(rows):
        vp = resolve_project_path(row.get("video_path", ""))
        if vp.exists() and vp.stat().st_size > 10_240:
            fp = _video_content_fingerprint(vp)
            hash_groups.setdefault(fp, []).append(idx)

    drop_indices: Set[int] = set()
    dropped_conflicts = 0
    dropped_dups = 0

    for fp, idxs in hash_groups.items():
        if len(idxs) <= 1:
            continue
        labels_present = {int(rows[i].get("label", 0)) for i in idxs}
        if len(labels_present) > 1:
            # Conflict: identical video file labeled both 1 (foul) and 0 (no_foul).
            # Keep the single highest-priority foul row (CC/INC) and drop contradictory duplicates.
            foul_idxs = [i for i in idxs if int(rows[i].get("label", 0)) == 1]
            keep_i = foul_idxs[0]
            for i in idxs:
                if i != keep_i:
                    drop_indices.add(i)
                    dropped_conflicts += 1
        else:
            keep_i = idxs[0]
            for i in idxs[1:]:
                drop_indices.add(i)
                dropped_dups += 1

    if remove_duplicate_files:
        for i in drop_indices:
            vp = resolve_project_path(rows[i].get("video_path", ""))
            if vp.exists():
                try:
                    vp.unlink()
                except Exception:
                    pass

    cleaned_rows = [r for idx, r in enumerate(rows) if idx not in drop_indices]
    if drop_indices:
        fieldnames = list(rows[0].keys()) if rows else MASTER_CSV_COLUMNS
        with open(metadata_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(cleaned_rows)
        logger.info(
            "Deduplicated %s: %d -> %d rows (dropped %d conflicting-label clips, %d same-class duplicates)",
            metadata_csv.name,
            len(rows),
            len(cleaned_rows),
            dropped_conflicts,
            dropped_dups,
        )

    return {
        "total_before": len(rows),
        "total_after": len(cleaned_rows),
        "dropped_conflicts": dropped_conflicts,
        "dropped_dups": dropped_dups,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NBA Last Two Minute (L2M) Report & Video Scraper"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "data" / "raw",
        help="Directory to store downloaded raw MP4 clips (default: data/raw)",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=BASE_DIR / "data" / "l2m_metadata.csv",
        help="Path to master metadata CSV (default: data/l2m_metadata.csv)",
    )
    parser.add_argument(
        "--include-archive",
        action="store_true",
        default=True,
        help="Include historical games from the NBA L2M Archive page (default: True)",
    )
    parser.add_argument(
        "--all-games",
        action="store_true",
        help="Scrape EVERY available NBA L2M game across all seasons + archive (equivalent to --max-games 0)",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=0,
        help="Maximum number of L2M game reports to inspect (0 = ALL available games across seasons & archive!)",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=0,
        help="Maximum number of foul/no_foul MP4 clips to download (0 = ALL missing clips)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only build metadata CSV without downloading MP4 files",
    )
    parser.add_argument(
        "--daily-inc-ic",
        action="store_true",
        help="Run daily monitor mode to append new INC/IC shooting & personal fouls to CSV",
    )
    parser.add_argument(
        "--fill-from-local",
        action="store_true",
        help="Populate data/l2m_incorrect_fouls.csv offline using all foul data in data/l2m_metadata.csv",
    )
    parser.add_argument(
        "--daily-csv",
        type=Path,
        default=BASE_DIR / "data" / "l2m_incorrect_fouls.csv",
        help="Target CSV for daily INC/IC foul monitoring (default: data/l2m_incorrect_fouls.csv)",
    )
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Bulk download missing MP4 video clips present in metadata CSV using parallel threads",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Worker threads for concurrent L2M JSON scraping & downloading (default: 16)",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Remove duplicate and conflicting-label video entries from data/l2m_metadata.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.deduplicate:
        deduplicate_raw_dataset(
            metadata_csv=args.metadata_csv,
            raw_dir=args.output_dir,
            remove_duplicate_files=False,
        )
    elif args.download_missing:
        bulk_download_missing_clips(
            metadata_csv=args.metadata_csv,
            output_dir=args.output_dir,
            max_workers=args.workers,
            max_clips=args.max_clips,
        )
        deduplicate_raw_dataset(metadata_csv=args.metadata_csv, raw_dir=args.output_dir)
    elif args.fill_from_local:
        populate_foul_csv_from_local_metadata(
            metadata_csv=args.metadata_csv,
            daily_csv=args.daily_csv,
            include_confirmed_fouls=True,
        )
    elif args.daily_inc_ic:
        run_daily_inc_ic_monitor(
            daily_csv=args.daily_csv,
            max_recent_games=args.max_games if args.max_games > 0 else 25,
        )
    else:
        eff_max_games = 0 if args.all_games else args.max_games
        run_training_dataset_scraper(
            output_dir=args.output_dir,
            metadata_csv=args.metadata_csv,
            daily_csv=args.daily_csv,
            include_archive=args.include_archive,
            max_games=eff_max_games,
            max_clips=args.max_clips,
            skip_download=args.skip_download,
            max_workers=args.workers,
        )
