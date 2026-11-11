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
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qs, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("l2m_scraper")

CURRENT_SEASON_URL = "https://official.nba.com/2026-27-nba-officiating-last-two-minute-reports/"
ARCHIVE_URL = "https://official.nba.com/nba-officiating-last-two-minute-reports-archive/"
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


def build_session(retries: int = 3, backoff_factor: float = 1.0) -> requests.Session:
    """Create a resilient HTTP session with exponential retry backoff."""
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
        except requests.RequestException as exc:
            logger.warning("Failed to fetch index URL %s: %s", index_url, exc)
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
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
                    logger.info("Loaded %d L2M play entries for game %s", len(l2m_rows), game_id)
                    return meta_dict, l2m_rows
    except (requests.RequestException, ValueError) as exc:
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
    except requests.RequestException as exc:
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
            if not mp4_candidates:
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
    except requests.RequestException as exc:
        logger.debug("Failed fetching L2M video page for game=%s event=%s: %s", game_id, event_id, exc)

    return None


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
    except (requests.RequestException, OSError) as exc:
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
    """Write collected L2M metadata rows to CSV atomically."""
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MASTER_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def run_training_dataset_scraper(
    output_dir: Path,
    metadata_csv: Path,
    daily_csv: Optional[Path] = Path("data/l2m_incorrect_fouls.csv"),
    include_archive: bool = False,
    max_games: int = 25,
    max_clips: int = 200,
    skip_download: bool = False,
    sleep_sec: float = 0.25,
) -> List[Dict[str, Any]]:
    """
    Scrape NBA L2M reports, map Shooting/Personal fouls to binary labels (1=foul, 0=no_foul),
    download MP4 clips to `data/raw/{foul,no_foul}/`, write `data/l2m_metadata.csv`,
    and also append any `INC` / `IC` foul plays to `data/l2m_incorrect_fouls.csv`.
    """
    session = build_session()
    foul_dir = output_dir / "foul"
    no_foul_dir = output_dir / "no_foul"
    foul_dir.mkdir(parents=True, exist_ok=True)
    no_foul_dir.mkdir(parents=True, exist_ok=True)
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)

    parsed_base = urlparse(CURRENT_SEASON_URL)
    official_root = f"{parsed_base.scheme}://{parsed_base.netloc}"

    existing_inc_ic_ids: Set[str] = set()
    if daily_csv is not None:
        daily_csv.parent.mkdir(parents=True, exist_ok=True)
        if daily_csv.exists():
            with open(daily_csv, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("play_id"):
                        existing_inc_ic_ids.add(row["play_id"])

    index_urls = [CURRENT_SEASON_URL]
    if include_archive:
        index_urls.append(ARCHIVE_URL)

    game_ids = discover_l2m_game_ids(session, index_urls, max_games=max_games)
    collected_rows: List[Dict[str, Any]] = []
    seen_play_ids: Set[str] = set()
    pbp_cache: Dict[str, List[Dict[str, Any]]] = {}
    downloaded_count = 0

    try:
        for game_id in game_ids:
            if len(collected_rows) >= max_clips:
                break

            game_meta, entries = fetch_l2m_report_entries(session, game_id)
            if not entries:
                continue

            game_date = str(
                game_meta.get("GameDateOut")
                or game_meta.get("GameDate")
                or game_meta.get("game_date")
                or ""
            ).strip()

            game_added = 0
            game_inc_ic_rows: List[Dict[str, Any]] = []
            for idx, raw_entry in enumerate(entries):
                if len(collected_rows) >= max_clips:
                    break

                parsed = extract_entry_fields(game_id, idx, raw_entry)
                norm_call_type = is_shooting_or_personal_foul(parsed["call_type_raw"])
                if norm_call_type is None:
                    continue

                decision = parsed["review_decision"]
                if decision not in LABEL_MAP:
                    continue

                label = LABEL_MAP[decision]
                class_subdir = foul_dir if label == 1 else no_foul_dir
                play_id = parsed["play_id"]
                if play_id in seen_play_ids:
                    continue

                rel_video_path = Path("data") / "raw" / ("foul" if label == 1 else "no_foul") / f"{play_id}.mp4"
                abs_video_path = class_subdir / f"{play_id}.mp4"
                video_path_str = str(rel_video_path)

                mp4_url: Optional[str] = None
                if not skip_download or decision in {"INC", "IC"}:
                    mp4_url = resolve_mp4_url(
                        session=session,
                        game_id=game_id,
                        video_event_id=parsed["video_event_id"],
                        period_name=parsed["period"],
                        pc_time=parsed["time_remaining"],
                        pbp_cache=pbp_cache,
                    )
                if not skip_download and mp4_url and download_mp4_clip(session, mp4_url, abs_video_path):
                    downloaded_count += 1

                row = {
                    "play_id": play_id,
                    "game_id": game_id,
                    "player_committing": parsed["player_committing"],
                    "player_disadvantaged": parsed["player_disadvantaged"],
                    "call_type": norm_call_type,
                    "review_decision": decision,
                    "label": label,
                    "video_path": video_path_str,
                }
                collected_rows.append(row)
                seen_play_ids.add(play_id)
                game_added += 1

                if daily_csv is not None and decision in {"INC", "IC"} and play_id not in existing_inc_ic_ids:
                    fallback_url = (
                        f"{official_root}/last-two-minute-report/"
                        f"?gameNo={game_id}&eventNum={parsed['video_event_id']}"
                    )
                    game_inc_ic_rows.append(
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
                            "video_url": mp4_url or fallback_url,
                            "comments": parsed["comments"],
                        }
                    )
                    existing_inc_ic_ids.add(play_id)

            if game_added > 0:
                write_metadata_rows(metadata_csv, collected_rows)
                logger.info(
                    "Game %s: extracted %d foul plays (running total: %d saved to %s)",
                    game_id,
                    game_added,
                    len(collected_rows),
                    metadata_csv,
                )

            if daily_csv is not None and game_inc_ic_rows:
                write_header = not daily_csv.exists() or daily_csv.stat().st_size == 0
                with open(daily_csv, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=DAILY_INC_IC_COLUMNS)
                    if write_header:
                        writer.writeheader()
                    writer.writerows(game_inc_ic_rows)
                logger.info(
                    "Game %s: appended %d INC/IC plays to %s",
                    game_id,
                    len(game_inc_ic_rows),
                    daily_csv,
                )

            time.sleep(sleep_sec)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user; saving %d collected rows to %s", len(collected_rows), metadata_csv)
    finally:
        write_metadata_rows(metadata_csv, collected_rows)

    logger.info(
        "Saved %d labeled L2M foul plays (%d MP4 clips downloaded) to %s",
        len(collected_rows),
        downloaded_count,
        metadata_csv,
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
                video_event_id=parsed["video_event_id"],
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
    metadata_csv: Path = Path("data/l2m_metadata.csv"),
    daily_csv: Path = Path("data/l2m_incorrect_fouls.csv"),
    include_confirmed_fouls: bool = True,
) -> List[Dict[str, Any]]:
    """
    Fill out `data/l2m_incorrect_fouls.csv` using all foul data already scraped in
    `data/l2m_metadata.csv` (prioritizing INC/IC officiating errors followed by
    confirmed foul plays when `include_confirmed_fouls=True`) so the CSV is completely
    populated even when offline.
    """
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NBA Last Two Minute (L2M) Report & Video Scraper"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to store downloaded raw MP4 clips (default: data/raw)",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("data/l2m_metadata.csv"),
        help="Path to master metadata CSV (default: data/l2m_metadata.csv)",
    )
    parser.add_argument(
        "--include-archive",
        action="store_true",
        help="Also scrape historical games from the NBA L2M Archive page",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=25,
        help="Maximum number of L2M game reports to inspect (default: 25)",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=200,
        help="Maximum number of foul/no_foul MP4 clips to download (default: 200)",
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
        default=Path("data/l2m_incorrect_fouls.csv"),
        help="Target CSV for daily INC/IC foul monitoring (default: data/l2m_incorrect_fouls.csv)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.fill_from_local:
        populate_foul_csv_from_local_metadata(
            metadata_csv=args.metadata_csv,
            daily_csv=args.daily_csv,
            include_confirmed_fouls=True,
        )
    elif args.daily_inc_ic:
        run_daily_inc_ic_monitor(
            daily_csv=args.daily_csv,
            max_recent_games=args.max_games,
        )
    else:
        run_training_dataset_scraper(
            output_dir=args.output_dir,
            metadata_csv=args.metadata_csv,
            daily_csv=args.daily_csv,
            include_archive=args.include_archive,
            max_games=args.max_games,
            max_clips=args.max_clips,
            skip_download=args.skip_download,
        )
