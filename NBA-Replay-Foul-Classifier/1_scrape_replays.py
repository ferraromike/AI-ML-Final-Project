#!/usr/bin/env python3
"""
STAGE 1: Official NBA Replay Archive Scraper & Multi-Challenge Dataset Builder (`1_scrape_replays.py`)

Scrapes official NBA Replay Center clips from:
  - Archive Portal : https://official.nba.com/replay/archive/
  - Replay Feed API: https://official.nba.com/wp-json/api/v1/query_replay_feed
  - Replay CDN     : https://videos.nba.com/nba/official-replay/media/...

Organizes raw MP4 clips and metadata by Challenge / Review Type (`data/raw/{foul,oob,goaltending,flagrant}/`):
  1. `challenge_type` (`challenge_type_id` in `0..3`):
       - `0 = foul`        : Coach's Challenge of Called Foul (`data/raw/foul/`)
       - `1 = oob`         : Coach's Challenge of Out of Bounds (`data/raw/oob/`)
       - `2 = goaltending` : Coach's Challenge of Goaltending / Basket Interference (`data/raw/goaltending/`)
       - `3 = flagrant`    : Review of Called Foul / Flagrant 1 & 2 / Clear-Path (`data/raw/flagrant/`)
  2. `outcome_category` / `call_label` (`0..9` — Full Challenge Resolution + Specific Correct Call):
       - `0 = foul_stands`                     : Called Foul Stands (Personal / Shooting Foul Upheld)
       - `1 = foul_overturned_no_foul`         : Called Foul Overturned to No Foul (Legal / Marginal Contact)
       - `2 = foul_overturned_other_foul`      : Called Foul Overturned to Offensive / Loose-Ball / Other Foul
       - `3 = oob_stands`                      : Out-of-Bounds Ruling Stands (Original Possession Upheld)
       - `4 = oob_overturned`                  : Out-of-Bounds Overturned (Possession Reversed / Jump Ball)
       - `5 = oob_proximate_foul`              : Out-of-Bounds Overturned due to Proximate Foul Assessed
       - `6 = goaltending_bi_stands`           : Goaltending / Basket Interference Ruling Stands
       - `7 = goaltending_bi_overturned`       : Goaltending / Basket Interference Overturned (Legal Play)
       - `8 = flagrant_confirmed`              : Flagrant Foul (Penalty 1 / 2) Confirmed
       - `9 = flagrant_downgraded_common_foul` : Common Personal Foul (Did Not Meet Flagrant Criteria)
  3. `label` (`0` or `1` binary foul indicator on the play):
       - `1`: A foul is the final call on the play (`foul_stands`, `foul_overturned_other_foul`, `oob_proximate_foul`, `flagrant_*`)
       - `0`: No foul on the play (`foul_overturned_no_foul`, `oob_stands`, `oob_overturned`, `goaltending_bi_*`)

Outputs:
  - `data/replay_metadata.csv`
  - `data/raw/foul/{play_id}.mp4`
  - `data/raw/oob/{play_id}.mp4`
  - `data/raw/goaltending/{play_id}.mp4`
  - `data/raw/flagrant/{play_id}.mp4`
"""

from __future__ import annotations

import argparse
import csv
import html as html_lib
import json
import logging
import re
import shutil
import threading
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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


BASE_DIR = Path(__file__).resolve().parent

REPLAY_ARCHIVE_URL = "https://official.nba.com/replay/archive/"
REPLAY_FEED_API_URL = "https://official.nba.com/wp-json/api/v1/query_replay_feed"

DEFAULT_SEASONS: Tuple[str, ...] = (
    "2026-27",
    "2025-26",
    "2024-25",
    "2023-24",
    "2022-23",
    "2021-22",
    "2020-21",
    "2019-20",
)

# Canonical Challenge / Review Folder Types (used directly as subfolder names in data/raw/ and data/processed/)
CHALLENGE_TYPES: Tuple[str, ...] = (
    "foul",        # 0: Coach's Challenge of a Called Personal/Shooting/Loose-Ball Foul
    "oob",         # 1: Coach's Challenge of an Out-of-Bounds Ruling
    "goaltending", # 2: Coach's Challenge of Goaltending / Basket Interference
    "flagrant",    # 3: Review of Called Foul (Flagrant 1 / Flagrant 2 / Common Foul / Clear-Path)
)
CHALLENGE_TYPE_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(CHALLENGE_TYPES)}
ID_TO_CHALLENGE_TYPE: Dict[int, str] = {idx: name for idx, name in enumerate(CHALLENGE_TYPES)}

CHALLENGE_TYPE_ALIASES: Dict[str, str] = {
    "foul": "foul",
    "challenge_foul": "foul",
    "called_foul": "foul",
    "challenge-of-called-foul": "foul",
    "oob": "oob",
    "challenge_oob": "oob",
    "out_of_bounds": "oob",
    "out-of-bounds": "oob",
    "challenge-oob": "oob",
    "goaltending": "goaltending",
    "goaltending_bi": "goaltending",
    "challenge_goaltending_bi": "goaltending",
    "basket_interference": "goaltending",
    "bi": "goaltending",
    "challenge-goaltending-bi": "goaltending",
    "flagrant": "flagrant",
    "challenge_flagrant": "flagrant",
    "flagrant_foul": "flagrant",
    "review-of-called-foul": "flagrant",
}

CHALLENGE_TYPE_DISPLAY: Dict[str, str] = {
    "foul": "CHALLENGE OF CALLED FOUL",
    "oob": "CHALLENGE OF OUT OF BOUNDS",
    "goaltending": "CHALLENGE OF GOALTENDING / BI",
    "flagrant": "FLAGRANT / CALLED FOUL REVIEW",
}

# Canonical 10-Class Full Challenge Resolution + Specific Call Taxonomy
CALL_CLASSES: Tuple[str, ...] = (
    "foul_stands",                     # 0
    "foul_overturned_no_foul",         # 1
    "foul_overturned_other_foul",      # 2
    "oob_stands",                      # 3
    "oob_overturned",                  # 4
    "oob_proximate_foul",              # 5
    "goaltending_bi_stands",           # 6
    "goaltending_bi_overturned",       # 7
    "flagrant_confirmed",              # 8
    "flagrant_downgraded_common_foul", # 9
)
CALL_CLASS_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(CALL_CLASSES)}
ID_TO_CALL_CLASS: Dict[int, str] = {idx: name for idx, name in enumerate(CALL_CLASSES)}

CALL_CLASS_SPECS: Dict[str, Dict[str, Any]] = {
    "foul_stands": {
        "call_label": 0,
        "challenge_type": "foul",
        "challenge_type_id": 0,
        "call_resolution": "CALL STANDS",
        "correct_call": "CALLED FOUL STANDS (Personal / Shooting Foul Upheld)",
        "short_verdict": "FOUL (CALL STANDS)",
        "is_foul": 1,
    },
    "foul_overturned_no_foul": {
        "call_label": 1,
        "challenge_type": "foul",
        "challenge_type_id": 0,
        "call_resolution": "CALL OVERTURNED",
        "correct_call": "NO FOUL (Overturned — Legal Guarding / Marginal Contact)",
        "short_verdict": "NO FOUL (CALL OVERTURNED)",
        "is_foul": 0,
    },
    "foul_overturned_other_foul": {
        "call_label": 2,
        "challenge_type": "foul",
        "challenge_type_id": 0,
        "call_resolution": "CALL OVERTURNED",
        "correct_call": "FOUL OVERTURNED TO OFFENSIVE / LOOSE-BALL FOUL",
        "short_verdict": "OFFENSIVE / OTHER FOUL (OVERTURNED)",
        "is_foul": 1,
    },
    "oob_stands": {
        "call_label": 3,
        "challenge_type": "oob",
        "challenge_type_id": 1,
        "call_resolution": "CALL STANDS",
        "correct_call": "OUT OF BOUNDS STANDS (Original Possession Upheld)",
        "short_verdict": "OUT OF BOUNDS (CALL STANDS)",
        "is_foul": 0,
    },
    "oob_overturned": {
        "call_label": 4,
        "challenge_type": "oob",
        "challenge_type_id": 1,
        "call_resolution": "CALL OVERTURNED",
        "correct_call": "OUT OF BOUNDS OVERTURNED (Possession Reversed)",
        "short_verdict": "OUT OF BOUNDS (CALL OVERTURNED)",
        "is_foul": 0,
    },
    "oob_proximate_foul": {
        "call_label": 5,
        "challenge_type": "oob",
        "challenge_type_id": 1,
        "call_resolution": "CALL OVERTURNED",
        "correct_call": "OUT OF BOUNDS OVERTURNED — PROXIMATE FOUL ASSESSED",
        "short_verdict": "PROXIMATE FOUL ON OOB (OVERTURNED)",
        "is_foul": 1,
    },
    "goaltending_bi_stands": {
        "call_label": 6,
        "challenge_type": "goaltending",
        "challenge_type_id": 2,
        "call_resolution": "CALL STANDS",
        "correct_call": "GOALTENDING / BASKET INTERFERENCE STANDS",
        "short_verdict": "GOALTENDING / BI (CALL STANDS)",
        "is_foul": 0,
    },
    "goaltending_bi_overturned": {
        "call_label": 7,
        "challenge_type": "goaltending",
        "challenge_type_id": 2,
        "call_resolution": "CALL OVERTURNED",
        "correct_call": "NO GOALTENDING / BI (Overturned — Legal Play / Clean Block)",
        "short_verdict": "NO GOALTENDING / BI (OVERTURNED)",
        "is_foul": 0,
    },
    "flagrant_confirmed": {
        "call_label": 8,
        "challenge_type": "flagrant",
        "challenge_type_id": 3,
        "call_resolution": "CALL STANDS",
        "correct_call": "FLAGRANT FOUL (PENALTY 1 / 2) CONFIRMED",
        "short_verdict": "FLAGRANT FOUL CONFIRMED",
        "is_foul": 1,
    },
    "flagrant_downgraded_common_foul": {
        "call_label": 9,
        "challenge_type": "flagrant",
        "challenge_type_id": 3,
        "call_resolution": "CALL OVERTURNED",
        "correct_call": "COMMON PERSONAL FOUL (Did Not Meet Flagrant Criteria)",
        "short_verdict": "COMMON FOUL (NOT FLAGRANT)",
        "is_foul": 1,
    },
}

LEGACY_OUTCOME_ALIASES: Dict[str, str] = {
    "foul_upheld": "foul_stands",
    "called_foul_review": "flagrant_confirmed",
    "flagrant_review_foul": "flagrant_confirmed",
    "overturned_to_no_foul": "foul_overturned_no_foul",
    "review_rescinded_no_foul": "foul_overturned_no_foul",
    "overturned_to_other_foul": "foul_overturned_other_foul",
    "proximate_foul_assessed": "oob_proximate_foul",
    "non_foul_challenge_clean": "oob_overturned",
}

CHALLENGE_TYPE_VALID_CALL_IDS: Dict[int, Tuple[int, ...]] = {
    0: (0, 1, 2),  # foul        -> foul_stands, foul_overturned_no_foul, foul_overturned_other_foul
    1: (3, 4, 5),  # oob         -> oob_stands, oob_overturned, oob_proximate_foul
    2: (6, 7),     # goaltending -> goaltending_bi_stands, goaltending_bi_overturned
    3: (8, 9),     # flagrant    -> flagrant_confirmed, flagrant_downgraded_common_foul
}

OFFICIAL_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/129.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html, application/xhtml+xml, */*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": REPLAY_ARCHIVE_URL,
    "Connection": "keep-alive",
}

METADATA_COLUMNS: List[str] = [
    "play_id",
    "archive_id",
    "ops_video_id",
    "game_id",
    "season",
    "game_date",
    "trigger",
    "challenge_type",
    "challenge_type_id",
    "title",
    "teams",
    "outcome_category",
    "call_label",
    "call_resolution",
    "correct_call",
    "ruling_text",
    "label",
    "video_path",
    "mp4_url",
    "permalink",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
logger = logging.getLogger("replay_scraper")


def resolve_project_path(path_arg: Path | str) -> Path:
    """Resolve relative paths consistently across OSes and execution directories."""
    norm_str = str(path_arg).replace("\\", "/")
    cand = Path(norm_str)
    if cand.is_absolute():
        return cand
    if cand.parts and cand.parts[0] == BASE_DIR.name:
        return (BASE_DIR.parent / cand).resolve()
    if cand.exists():
        return cand.resolve()
    return (BASE_DIR / cand).resolve()


def normalize_challenge_type(raw_ctype: Any) -> str:
    """Map any challenge type string or alias to one of `('foul', 'oob', 'goaltending', 'flagrant')`."""
    s = str(raw_ctype or "").strip().lower()
    return CHALLENGE_TYPE_ALIASES.get(s, "foul")


class _StdlibResponse:
    def __init__(self, status_code: int, content: bytes, stream_resp: Any = None) -> None:
        self.status_code = status_code
        self.content = content
        self._stream_resp = stream_resp

    @property
    def text(self) -> str:
        return self.content.decode("utf-8-sig", errors="replace")

    def json(self) -> Any:
        return json.loads(self.text)

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
    def get(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: int = 20,
        stream: bool = False,
    ) -> _StdlibResponse:
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


_SHARED_HTML_SESSION: Any = None
_SHARED_CDN_SESSION: Any = None
_SESSION_LOCK = threading.Lock()


def build_session(retries: int = 2, backoff_factor: float = 0.4, pool_maxsize: int = 16) -> Any:
    """Create a resilient HTTP session with keep-alive connection pooling and quiet retry backoff."""
    if not HAS_REQUESTS or requests is None:
        return _StdlibSession()
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=pool_maxsize, pool_maxsize=pool_maxsize)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_thread_session(for_html: bool = False) -> Any:
    """
    Return a shared connection-pooled session so worker threads reuse persistent
    keep-alive TLS sockets instead of opening new handshakes on every batch.
    """
    global _SHARED_HTML_SESSION, _SHARED_CDN_SESSION
    with _SESSION_LOCK:
        if for_html:
            if _SHARED_HTML_SESSION is None:
                _SHARED_HTML_SESSION = build_session(retries=1, backoff_factor=0.3, pool_maxsize=8)
            return _SHARED_HTML_SESSION
        if _SHARED_CDN_SESSION is None:
            _SHARED_CDN_SESSION = build_session(retries=2, backoff_factor=0.5, pool_maxsize=20)
        return _SHARED_CDN_SESSION


def normalize_game_id(raw_game_id: Any) -> str:
    """Zero-pad numeric NBA game IDs to standard 10-digit strings (e.g. '0022600257')."""
    digits = re.sub(r"\D", "", str(raw_game_id or ""))
    return digits.zfill(10) if digits else "0000000000"


def clean_bracket_list(raw_val: Any) -> str:
    """Convert Replay Feed bracket strings like '[utah-jazz, la-clippers]' to 'utah-jazz, la-clippers'."""
    s = str(raw_val or "").strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    return s


def make_play_id(entry: Dict[str, Any]) -> str:
    """Build a stable, filesystem-safe unique play_id from feed metadata."""
    game_id = normalize_game_id(entry.get("game_id"))
    ops_id = re.sub(r"\D", "", str(entry.get("opsreporting_video_id") or ""))
    archive_id = re.sub(r"\D", "", str(entry.get("id") or ""))
    suffix = ops_id or archive_id or "0"
    return f"{game_id}_{suffix}"


def extract_cdn_media_slug(url_or_name: str) -> Optional[str]:
    """
    Extract canonical NBA Replay CDN media slug (e.g. `trigger16_phxsac_1_111826_approved`)
    from an MP4 URL, thumbnail URL, or local video filename.
    """
    s = str(url_or_name or "").strip().lower()
    m = re.search(r"(trigger\d+_[a-z0-9]+_\d+_\d+(?:_approved)?)", s)
    return m.group(1) if m else None


def derive_mp4_from_thumbnail(thumbnail_url: str, quality: str = "hd") -> Optional[str]:
    """
    Derive direct MP4 URL from an NBA Replay Archive thumbnail URL, e.g.:
      .../trigger16_utalac_1_111726_approved_320x180.0000000.jpg
      -> .../trigger16_utalac_1_111726_approved_1280x720.mp4
    """
    thumb = str(thumbnail_url or "").strip()
    if not thumb or "videos.nba.com/nba/official-replay/media/" not in thumb:
        return None
    target_res = "1280x720" if quality == "hd" else "320x180"
    replaced = re.sub(
        r"_(?:1280x720|320x180)\.\d+\.jpg$",
        f"_{target_res}.mp4",
        thumb,
        flags=re.IGNORECASE,
    )
    return replaced if replaced.endswith(".mp4") else None


def find_existing_raw_mp4(play_id: str, output_dir: Path = BASE_DIR / "data" / "raw") -> Optional[Path]:
    """Locate an already-downloaded raw MP4 for `play_id` across all challenge folders (including legacy `no_foul`)."""
    out_dir = resolve_project_path(output_dir)
    for folder in (*CHALLENGE_TYPES, "no_foul"):
        cand = out_dir / folder / f"{play_id}.mp4"
        if cand.exists() and cand.stat().st_size > 10_240:
            return cand
    return None


def migrate_raw_clips_to_challenge_folders(
    rows: List[Dict[str, Any]],
    output_dir: Path = BASE_DIR / "data" / "raw",
) -> int:
    """
    Ensure all downloaded MP4 clips in `data/raw/` live inside their challenge-type folder
    (`data/raw/{foul,oob,goaltending,flagrant}/{play_id}.mp4`) and clean up empty legacy `no_foul/`.
    """
    out_dir = resolve_project_path(output_dir)
    for ctype in CHALLENGE_TYPES:
        (out_dir / ctype).mkdir(parents=True, exist_ok=True)

    moved = 0
    for r in rows:
        ur = upgrade_metadata_row(r)
        pid = str(ur.get("play_id") or "").strip()
        ctype = normalize_challenge_type(ur.get("challenge_type"))
        if not pid:
            continue
        target_p = out_dir / ctype / f"{pid}.mp4"
        if target_p.exists() and target_p.stat().st_size > 10_240:
            continue
        existing_p = find_existing_raw_mp4(pid, out_dir)
        if existing_p is not None and existing_p.resolve() != target_p.resolve():
            target_p.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(existing_p), str(target_p))
            moved += 1

    legacy_no_foul = out_dir / "no_foul"
    if legacy_no_foul.exists():
        remaining = list(legacy_no_foul.glob("*.mp4"))
        for rem in remaining:
            dest = out_dir / "foul" / rem.name
            if not dest.exists():
                shutil.move(str(rem), str(dest))
                moved += 1
            else:
                rem.unlink(missing_ok=True)
        try:
            legacy_no_foul.rmdir()
        except OSError:
            pass
    return moved


def query_replay_feed(
    session: Any,
    season: str,
    trigger: str = "coachs-challenge",
    max_items: int = 0,
    sleep_sec: float = 0.20,
) -> List[Dict[str, Any]]:
    """
    Fetch replay archive feed items for a `(season, trigger)` slice from
    `/wp-json/api/v1/query_replay_feed`, handling pagination safely.
    """
    collected: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    offset = 0
    page_guard = 0

    while page_guard < 15:
        page_guard += 1
        params: Dict[str, Any] = {
            "offset": offset,
            "season": season,
            "trigger": trigger,
            "outcome": 0,
        }
        try:
            resp = session.get(
                REPLAY_FEED_API_URL,
                headers=OFFICIAL_HEADERS,
                params=params,
                timeout=25,
            )
            resp.raise_for_status()
            batch = json.loads(resp.content.decode("utf-8-sig", errors="replace"))
        except Exception as exc:
            logger.warning(
                "Feed query failed (season=%s, trigger=%s, offset=%d): %s",
                season,
                trigger,
                offset,
                exc,
            )
            break

        if not isinstance(batch, list) or not batch:
            break

        added_this_page = 0
        for item in batch:
            if not isinstance(item, dict):
                continue
            uid = str(item.get("id") or item.get("opsreporting_video_id") or item.get("permalink") or "")
            if not uid or uid in seen_ids:
                continue
            seen_ids.add(uid)
            item_copy = dict(item)
            item_copy["_queried_season"] = season
            collected.append(item_copy)
            added_this_page += 1
            if max_items > 0 and len(collected) >= max_items:
                return collected

        if added_this_page == 0 or len(batch) < 1000:
            break
        offset += len(batch)
        time.sleep(sleep_sec)

    return collected


def fetch_embedded_archive_feed(session: Any) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Fallback parser that reads the embedded `var replayFeed = [...]` JSON directly
    from `https://official.nba.com/replay/archive/`.
    """
    try:
        resp = session.get(REPLAY_ARCHIVE_URL, headers=OFFICIAL_HEADERS, timeout=25)
        resp.raise_for_status()
        raw = resp.text
        season_m = re.search(r"nbaSeason\s*=\s*['\"]([^'\"]+)['\"]", raw)
        season = season_m.group(1) if season_m else DEFAULT_SEASONS[0]
        feed_m = re.search(r"var\s+replayFeed\s*=\s*(\[.*?\]);", raw, flags=re.S)
        if not feed_m:
            return season, []
        items = json.loads(feed_m.group(1))
        if not isinstance(items, list):
            return season, []
        for it in items:
            if isinstance(it, dict):
                it["_queried_season"] = season
        return season, items
    except Exception as exc:
        logger.warning("Embedded replay archive fallback failed: %s", exc)
        return DEFAULT_SEASONS[0], []


def parse_permalink_html(raw_html: str, quality: str = "hd") -> Tuple[str, Optional[str]]:
    """
    Extract `(ruling_text, mp4_url)` from a single NBA Replay Archive permalink HTML page.
    Works with BeautifulSoup when available and falls back cleanly to regex parsing.
    """
    ruling_text = ""
    mp4_candidates: List[str] = []

    if HAS_BS4 and BeautifulSoup is not None:
        soup = BeautifulSoup(raw_html, "html.parser")
        details_div = soup.find("div", class_=lambda c: c and "trigger-details" in str(c))
        if details_div is not None:
            p_tags = []
            for p in details_div.find_all("p"):
                cls = " ".join(p.get("class") or [])
                if "video-date" in cls:
                    continue
                if p.find_parent("h2") is not None:
                    continue
                txt = re.sub(r"\s+", " ", p.get_text(" ", strip=True)).strip()
                if txt and "Game Box Score" not in txt:
                    p_tags.append(txt)
            ruling_text = " ".join(p_tags).strip()
            if not ruling_text:
                ruling_text = re.sub(r"\s+", " ", details_div.get_text(" ", strip=True)).strip()

        video_tag = soup.find("video", id="replay-video__player") or soup.find("video")
        if video_tag is not None:
            for src_tag in video_tag.find_all("source"):
                src = str(src_tag.get("src") or "").strip()
                if ".mp4" in src.lower():
                    mp4_candidates.append(src)

        og_video = soup.find("meta", attrs={"property": "og:video"})
        if og_video and og_video.get("content"):
            content_url = str(og_video["content"]).strip()
            if ".mp4" in content_url.lower():
                mp4_candidates.append(content_url)

    if not ruling_text:
        m_div = re.search(
            r'<div[^>]*class=["\'][^"\']*trigger-details[^"\']*["\'][^>]*>(.*?)</div>',
            raw_html,
            flags=re.S | re.I,
        )
        if m_div:
            block = m_div.group(1)
            block = re.sub(r"<h2[^>]*>.*?</h2>", " ", block, flags=re.S | re.I)
            block = re.sub(r'<p[^>]*class=["\'][^"\']*video-date[^"\']*["\'][^>]*>.*?</p>', " ", block, flags=re.S | re.I)
            txt = re.sub(r"<[^>]+>", " ", block)
            ruling_text = re.sub(r"\s+", " ", html_lib.unescape(txt)).strip()

    if not mp4_candidates:
        mp4_candidates = re.findall(
            r"https?://[^\s'\"<>]+\.mp4(?:\?[^\s'\"<>]*)?",
            raw_html,
            flags=re.I,
        )

    unique_mp4s = list(dict.fromkeys(u.replace("\\/", "/").strip() for u in mp4_candidates if u))
    selected_mp4: Optional[str] = None
    if unique_mp4s:
        if quality == "sd":
            for u in unique_mp4s:
                if "320x180" in u or "sd" in u.lower():
                    selected_mp4 = u
                    break
        else:
            for u in unique_mp4s:
                if "1280x720" in u or "hd" in u.lower() or "720" in u:
                    selected_mp4 = u
                    break
        if selected_mp4 is None:
            selected_mp4 = unique_mp4s[0]

    return ruling_text, selected_mp4


_OVERTURNED_TO_OTHER_FOUL_PATTERNS = (
    re.compile(
        r"(?:was\s*)?overturned\s+(?:and\s+(?:changed|ruled)\s+)?to\s+(?:a|an)\s+"
        r"(?:[\w\-]+\s+){0,4}foul\b",
        re.I,
    ),
    re.compile(
        r"(?:changed|modified|upgraded|downgraded|reclassified|assessed)\s+to\s+(?:a|an)\s+"
        r"(?:[\w\-]+\s+){0,4}foul\b",
        re.I,
    ),
    re.compile(
        r"(?:was\s*)?overturned[^.]{0,90}\b(?:defensive|offensive|personal|shooting|loose[\s\-]ball|"
        r"blocking|common|flagrant|away[\s\-]from[\s\-](?:the[\s\-])?play|take|transition|clear[\s\-]path)\s+foul\b",
        re.I,
    ),
    re.compile(
        r"\bproximate\s+foul\b|\b(?:was\s+|is\s+)?assessed\s+a\s+(?:[\w\-]+\s+){0,3}foul\b|\bfoul\s+(?:was\s+)?(?:called|assessed|committed)\s+on\b",
        re.I,
    ),
)

_OVERTURNED_TO_NO_FOUL_PATTERNS = (
    re.compile(r"(?:was\s*)?overturned\s+to\s+(?:a\s+)?no[\s\-]?(?:foul|call)\b", re.I),
    re.compile(r"\bno\s+foul\s+(?:occurred|was\s+committed|on\s+the\s+play)\b", re.I),
    re.compile(r"\b(?:ruled|determined)\s+(?:there\s+was\s+)?no\s+foul\b", re.I),
    re.compile(r"(?:was\s*)?overturned\s+to\s+(?:an?\s+)?inadvertent\s+whistle\b", re.I),
    re.compile(r"(?:was\s*)?overturned\s+(?:and\s+a\s+|to\s+a\s+)?jump\s+ball\b", re.I),
    re.compile(r"(?:was\s*)?overturned\s+to\s+(?:an?\s+)?out[\s\-]of[\s\-]bounds\b", re.I),
    re.compile(r"(?:was\s*)?overturned[^.]{0,70}\b(?:legal\s+guarding|incidental\s+contact|marginal\s+contact|clean\s+block|no\s+illegal\s+contact)\b", re.I),
    re.compile(r"\bchallenge\s+(?:was\s+)?successful\b[^.]{0,80}\bno\s+foul\b", re.I),
    re.compile(r"\bfoul\b[^.]{0,60}\b(?:was\s+)?rescinded\b", re.I),
)

_CALL_STANDS_PATTERNS = (
    re.compile(r"would\s*stand\b|\b(?:call|ruling)\s*(?:on\s*the\s*(?:floor|court)\s*)?stands\b", re.I),
    re.compile(r"\b(?:was|is)\s*(?:upheld|confirmed)\b", re.I),
    re.compile(r"\bchallenge\s*(?:was|is)\s*unsuccessful\b", re.I),
    re.compile(r"\bremains\s+a\s+(?:[\w\-]+\s+)?foul\b", re.I),
    re.compile(r"\bfoul\s+(?:was\s+)?(?:upheld|confirmed|stands|stood)\b", re.I),
)

_ANY_OVERTURNED_PATTERN = re.compile(r"(?:was\s*)?overturned|\bchallenge\s+(?:was\s+)?successful\b|\brescinded\b", re.I)


def detect_challenge_type(title: str, trigger: str, ruling_text: str = "", mp4_url: str = "") -> Optional[str]:
    """
    Determine the canonical challenge/review category (`'foul'`, `'oob'`, `'goaltending'`, or `'flagrant'`)
    from archive title, trigger slug, ruling text, or CDN trigger prefix.
    """
    title_low = (title or "").strip().lower()
    trigger_low = clean_bracket_list(trigger).lower()
    text_low = re.sub(r"\s+", " ", (ruling_text or "").strip()).lower()
    url_low = (mp4_url or "").strip().lower()

    # 1. Flagrant / Called Foul Referee Review
    if (
        "review of called foul" in title_low
        or "review-of-called-foul" in trigger_low
        or "foul-flagrant" in text_low
        or "flagrant foul" in title_low
        or "clear-path" in title_low
        or "clear-path" in trigger_low
        or "off-ball foul" in title_low
        or "off-ball-foul" in trigger_low
        or "trigger3_" in url_low
        or "trigger8_" in url_low
    ):
        return "flagrant"

    # 2. Goaltending / Basket Interference Challenge
    if (
        "challenge goaltending" in title_low
        or "challenge-goaltending-bi" in trigger_low
        or "goaltending/bi challenge" in text_low
        or "goaltending challenge" in text_low
        or "basket interference challenge" in text_low
        or "goaltending" in title_low
        or "basket interference" in title_low
        or "trigger18_" in url_low
    ):
        return "goaltending"

    # 3. Out-of-Bounds (OOB) Challenge
    if (
        "challenge oob" in title_low
        or "challenge-oob" in trigger_low
        or "out of bounds challenge" in text_low
        or "out-of-bounds challenge" in text_low
        or "out of bounds" in title_low
        or "trigger17_" in url_low
    ):
        return "oob"

    # 4. Coach's Challenge of Called Foul
    if (
        "challenge of called foul" in title_low
        or "foul challenge" in text_low
        or "trigger16_" in url_low
        or ("coachs-challenge" in trigger_low and "foul" in text_low)
    ):
        return "foul"

    # Fallback inspection of ruling text
    if "flagrant" in text_low and "foul challenge" not in text_low:
        return "flagrant"
    if "out of bounds" in text_low or "’s ball" in text_low or "s’ ball" in text_low or "'s ball" in text_low:
        return "oob"
    if "goaltending" in text_low or "basket interference" in text_low:
        return "goaltending"
    if "foul" in text_low:
        return "foul"

    return None


def classify_replay_ruling(
    title: str,
    trigger: str,
    ruling_text: str,
    include_flagrant_reviews: bool = True,
    include_non_foul_challenges: bool = True,
    mp4_url: str = "",
) -> Optional[Tuple[int, str]]:
    """
    Parse an NBA Replay Archive entry + verbatim ruling text into `(label, outcome_category)`:
      - `outcome_category` is one of the 10 canonical `CALL_CLASSES`:
          0: `foul_stands`                     (challenge_type='foul',        label=1)
          1: `foul_overturned_no_foul`         (challenge_type='foul',        label=0)
          2: `foul_overturned_other_foul`      (challenge_type='foul',        label=1)
          3: `oob_stands`                      (challenge_type='oob',         label=0)
          4: `oob_overturned`                  (challenge_type='oob',         label=0)
          5: `oob_proximate_foul`              (challenge_type='oob',         label=1)
          6: `goaltending_bi_stands`           (challenge_type='goaltending', label=0)
          7: `goaltending_bi_overturned`       (challenge_type='goaltending', label=0)
          8: `flagrant_confirmed`              (challenge_type='flagrant',    label=1)
          9: `flagrant_downgraded_common_foul` (challenge_type='flagrant',    label=1)
    """
    text_low = re.sub(r"\s+", " ", (ruling_text or "").strip()).lower()
    if not text_low:
        return None

    challenge_type = detect_challenge_type(title=title, trigger=trigger, ruling_text=ruling_text, mp4_url=mp4_url)
    if challenge_type is None:
        return None

    if challenge_type in ("oob", "goaltending") and not include_non_foul_challenges:
        return None
    if challenge_type == "flagrant" and not include_flagrant_reviews:
        return None

    has_other_foul = any(p.search(text_low) for p in _OVERTURNED_TO_OTHER_FOUL_PATTERNS)
    has_no_foul = any(p.search(text_low) for p in _OVERTURNED_TO_NO_FOUL_PATTERNS)
    has_stands = any(p.search(text_low) for p in _CALL_STANDS_PATTERNS)
    has_overturned = bool(_ANY_OVERTURNED_PATTERN.search(text_low))

    # --- CASE 1: Coach's Challenge of Called Foul ('foul') ---
    if challenge_type == "foul":
        if has_other_foul and not has_no_foul:
            return 1, "foul_overturned_other_foul"
        if has_no_foul and not has_other_foul:
            return 0, "foul_overturned_no_foul"
        if has_stands and not has_overturned:
            return 1, "foul_stands"
        if has_stands and has_overturned and not has_no_foul and not has_other_foul:
            return 1, "foul_stands"
        if has_overturned and not has_other_foul:
            if not re.search(r"overturned[^.]{0,65}\bfoul\s+on\b", text_low):
                return 0, "foul_overturned_no_foul"
            return 1, "foul_overturned_other_foul"
        return None

    # --- CASE 2: Coach's Challenge of Out of Bounds ('oob') ---
    if challenge_type == "oob":
        if has_other_foul or re.search(
            r"\b(?:assessed\s+a\s+(?:[\w\-]+\s+){0,3}foul|proximate\s+foul|offensive\s+foul|defensive\s+foul|loose[\s\-]ball\s+foul|personal\s+foul)\b",
            text_low,
        ):
            if "no foul" not in text_low:
                return 1, "oob_proximate_foul"
        if has_stands and not has_overturned:
            return 0, "oob_stands"
        if has_overturned:
            return 0, "oob_overturned"
        if has_stands:
            return 0, "oob_stands"
        return None

    # --- CASE 3: Coach's Challenge of Goaltending / Basket Interference ('goaltending') ---
    if challenge_type == "goaltending":
        if re.search(
            r"goaltending\s+was\s*overturned\s+to\s+(?:a\s+)?basket\s+interference|"
            r"basket\s+interference\s+was\s*overturned\s+to\s+(?:a\s+)?goaltending",
            text_low,
        ):
            return 0, "goaltending_bi_stands"
        if has_stands and not has_overturned:
            return 0, "goaltending_bi_stands"
        if has_overturned:
            return 0, "goaltending_bi_overturned"
        if has_stands:
            return 0, "goaltending_bi_stands"
        return None

    # --- CASE 4: Review of Called Foul / Flagrant ('flagrant') ---
    if challenge_type == "flagrant":
        if re.search(r"did\s*not\s+meet\s+the\s+criteria\s+for\s+a\s+flagrant|remains\s+a\s+common\s+foul|ruled\s+a\s+common\s+foul", text_low):
            return 1, "flagrant_downgraded_common_foul"
        if re.search(r"did\s+meet\s+the\s+criteria\s+for\s+a\s+flagrant|flagrant\s+foul\s+penalty|upgraded\s+to\s+a\s+flagrant|clear[\s\-]path", text_low):
            return 1, "flagrant_confirmed"
        if has_no_foul:
            return 0, "foul_overturned_no_foul"
        if "flagrant" in text_low or "common foul" in text_low or "personal foul" in text_low:
            return 1, "flagrant_confirmed"
        return None

    return None


def upgrade_metadata_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Upgrade or normalize a metadata dictionary so it always contains canonical challenge-type
    folder paths (`data/raw/{foul,oob,goaltending,flagrant}/{play_id}.mp4`) and call resolution columns.
    """
    out = dict(row)
    title = str(out.get("title") or "")
    trigger = str(out.get("trigger") or "")
    ruling_text = str(out.get("ruling_text") or "")
    mp4_url = str(out.get("mp4_url") or "")
    raw_cat = str(out.get("outcome_category") or "").strip()

    canonical_cat = LEGACY_OUTCOME_ALIASES.get(raw_cat, raw_cat)
    if canonical_cat not in CALL_CLASS_SPECS and ruling_text:
        classified = classify_replay_ruling(
            title=title,
            trigger=trigger,
            ruling_text=ruling_text,
            include_flagrant_reviews=True,
            include_non_foul_challenges=True,
            mp4_url=mp4_url,
        )
        if classified is not None:
            _, canonical_cat = classified

    if canonical_cat not in CALL_CLASS_SPECS:
        lbl = int(out.get("label", 0) or 0)
        ctype = detect_challenge_type(title, trigger, ruling_text, mp4_url) or "foul"
        if ctype == "oob":
            canonical_cat = "oob_proximate_foul" if lbl == 1 else "oob_overturned"
        elif ctype == "goaltending":
            canonical_cat = "goaltending_bi_overturned"
        elif ctype == "flagrant":
            canonical_cat = "flagrant_confirmed"
        else:
            canonical_cat = "foul_stands" if lbl == 1 else "foul_overturned_no_foul"

    spec = CALL_CLASS_SPECS[canonical_cat]
    ctype = spec["challenge_type"]
    out["outcome_category"] = canonical_cat
    out["challenge_type"] = ctype
    out["challenge_type_id"] = int(spec["challenge_type_id"])
    out["call_label"] = int(spec["call_label"])
    out["call_resolution"] = spec["call_resolution"]
    out["correct_call"] = spec["correct_call"]
    out["label"] = int(spec["is_foul"])

    play_id = str(out.get("play_id") or "").strip()
    if play_id:
        out["video_path"] = (Path("data") / "raw" / ctype / f"{play_id}.mp4").as_posix()
    return out


def harvest_rss_ruling_cache(
    session: Any,
    max_pages_per_query: int = 25,
    workers: int = 4,
    include_flagrant_reviews: bool = True,
    include_non_foul_challenges: bool = True,
) -> Dict[str, str]:
    """
    Bulk-harvest verbatim Replay Center rulings via WordPress RSS search feeds
    (`https://official.nba.com/archive/feed/?s=<query>&paged=N`), returning 10 full
    rulings per 10 KB XML request (~10x faster and 15x lighter than individual permalink HTML pages).
    Returns a dict mapping both normalized `permalink` and `id:<archive_id>` to `ruling_text`.
    """
    queries = ["Challenge of Called Foul"]
    if include_non_foul_challenges:
        queries.extend(["Challenge OOB", "Challenge Goaltending"])
    if include_flagrant_reviews:
        queries.append("Review of Called Foul")

    cache: Dict[str, str] = {}
    if max_pages_per_query <= 0:
        return cache

    def _fetch_rss_page(q_str: str, page_num: int) -> List[Tuple[str, str, str]]:
        sess = get_thread_session(for_html=True)
        url = "https://official.nba.com/archive/feed/"
        try:
            resp = sess.get(
                url,
                headers=OFFICIAL_HEADERS,
                params={"s": q_str, "paged": page_num},
                timeout=12,
            )
            if resp.status_code >= 400 or "<item>" not in resp.text:
                return []
            found: List[Tuple[str, str, str]] = []
            for item_m in re.finditer(r"<item>(.*?)</item>", resp.text, flags=re.S | re.I):
                block = item_m.group(1)
                link_m = re.search(r"<link>\s*(https?://[^\s<]+)\s*</link>", block, flags=re.I)
                guid_m = re.search(r"[?&]p=(\d+)", block, flags=re.I)
                desc_m = re.search(r"<description><!\[CDATA\[(.*?)\]\]></description>", block, flags=re.S | re.I)
                if desc_m:
                    raw_desc = re.sub(r"<[^>]+>", " ", desc_m.group(1))
                    clean_txt = re.sub(r"\s+", " ", html_lib.unescape(raw_desc)).strip()
                    link = link_m.group(1).strip().rstrip("/") if link_m else ""
                    aid = guid_m.group(1).strip() if guid_m else ""
                    if clean_txt and (link or aid):
                        found.append((link, aid, clean_txt))
            return found
        except Exception:
            return []

    tasks = [(q, p) for p in range(1, max_pages_per_query + 1) for q in queries]
    with ThreadPoolExecutor(max_workers=min(max(1, workers), 4)) as pool:
        futures = [pool.submit(_fetch_rss_page, q, p) for q, p in tasks]
        for fut in as_completed(futures):
            for link, aid, txt in fut.result():
                if link:
                    cache[link] = txt
                if aid:
                    cache[f"id:{aid}"] = txt
    return cache


def fetch_replay_details(
    entry: Dict[str, Any],
    quality: str = "hd",
    include_flagrant_reviews: bool = True,
    include_non_foul_challenges: bool = True,
    rss_cache: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Resolve an archive entry's ruling text from the bulk RSS cache (instant, 0 HTTP calls)
    or fall back to fetching its permalink HTML, and return a normalized metadata dict.
    """
    permalink = str(entry.get("permalink") or "").strip()
    if not permalink:
        return None

    norm_link = permalink.rstrip("/")
    archive_id = str(entry.get("id") or "").strip()
    cache = rss_cache or {}
    ruling_text = cache.get(norm_link) or (cache.get(f"id:{archive_id}") if archive_id else "") or ""
    mp4_url = derive_mp4_from_thumbnail(str(entry.get("thumbnail_url") or ""), quality=quality)

    if not ruling_text or not mp4_url:
        session = get_thread_session(for_html=True)
        try:
            resp = session.get(permalink, headers=OFFICIAL_HEADERS, timeout=12)
            resp.raise_for_status()
            parsed_text, parsed_mp4 = parse_permalink_html(resp.text, quality=quality)
            if parsed_text:
                ruling_text = parsed_text
            if parsed_mp4:
                mp4_url = parsed_mp4
        except Exception as exc:
            logger.debug("Skipped slow/unreachable permalink %s: %s", permalink, exc)
            return None

    if not mp4_url:
        return None

    title = html_lib.unescape(str(entry.get("title") or "").strip())
    trigger = clean_bracket_list(entry.get("trigger"))
    classified = classify_replay_ruling(
        title=title,
        trigger=trigger,
        ruling_text=ruling_text,
        include_flagrant_reviews=include_flagrant_reviews,
        include_non_foul_challenges=include_non_foul_challenges,
        mp4_url=mp4_url,
    )
    if classified is None:
        return None

    label, outcome_category = classified
    spec = CALL_CLASS_SPECS[outcome_category]
    ctype = spec["challenge_type"]
    play_id = make_play_id(entry)
    game_id = normalize_game_id(entry.get("game_id"))
    rel_video_path = (Path("data") / "raw" / ctype / f"{play_id}.mp4").as_posix()

    return {
        "play_id": play_id,
        "archive_id": str(entry.get("id") or ""),
        "ops_video_id": str(entry.get("opsreporting_video_id") or ""),
        "game_id": game_id,
        "season": str(entry.get("_queried_season") or entry.get("season") or ""),
        "game_date": str(entry.get("game_date") or entry.get("date") or ""),
        "trigger": trigger,
        "challenge_type": ctype,
        "challenge_type_id": int(spec["challenge_type_id"]),
        "title": title,
        "teams": clean_bracket_list(entry.get("teams")),
        "outcome_category": outcome_category,
        "call_label": int(spec["call_label"]),
        "call_resolution": spec["call_resolution"],
        "correct_call": spec["correct_call"],
        "ruling_text": ruling_text,
        "label": int(label),
        "video_path": rel_video_path,
        "mp4_url": mp4_url,
        "permalink": permalink,
    }


def download_mp4_clip(
    mp4_url: str,
    dest_path: Path,
    chunk_size: int = 65536,
) -> bool:
    """
    Stream-download a single MP4 replay clip safely using atomic `.part` rename.
    Automatically falls back between `_1280x720.mp4` and `_320x180.mp4` if one resolution is unavailable on CDN.
    """
    if dest_path.exists() and dest_path.stat().st_size > 10_240:
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(".mp4.part")
    session = get_thread_session(for_html=False)

    candidate_urls = [mp4_url]
    if "_1280x720.mp4" in mp4_url:
        candidate_urls.append(mp4_url.replace("_1280x720.mp4", "_320x180.mp4"))
    elif "_320x180.mp4" in mp4_url:
        candidate_urls.append(mp4_url.replace("_320x180.mp4", "_1280x720.mp4"))

    for cand_url in candidate_urls:
        try:
            with session.get(cand_url, headers=OFFICIAL_HEADERS, stream=True, timeout=25) as resp:
                if resp.status_code >= 400:
                    continue
                with open(tmp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
            if tmp_path.exists() and tmp_path.stat().st_size > 10_240:
                tmp_path.replace(dest_path)
                return True
            tmp_path.unlink(missing_ok=True)
        except Exception as exc:
            logger.debug("Failed downloading %s: %s", cand_url, exc)
            tmp_path.unlink(missing_ok=True)
    return False


def load_existing_metadata(metadata_csv: Path) -> List[Dict[str, Any]]:
    if not metadata_csv.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(metadata_csv, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("play_id"):
                rows.append(upgrade_metadata_row(dict(row)))
    return rows


def write_metadata_rows(metadata_csv: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows and metadata_csv.exists() and metadata_csv.stat().st_size > 100:
        return
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = metadata_csv.with_suffix(".csv.tmp")
    upgraded = [upgrade_metadata_row(r) for r in rows]
    with open(tmp_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(upgraded)
    tmp_csv.replace(metadata_csv)


def interleave_balanced_rows(rows: List[Dict[str, Any]], max_total: int = 0) -> List[Dict[str, Any]]:
    """
    Round-robin interleave rows across all `outcome_category` call classes across
    `foul`, `oob`, `goaltending`, and `flagrant` so any prefix contains a balanced mix.
    """
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        ur = upgrade_metadata_row(r)
        buckets[str(ur.get("outcome_category", "foul_stands"))].append(ur)

    active_keys = [k for k in CALL_CLASSES if buckets.get(k)]
    for k in buckets:
        if k not in active_keys:
            active_keys.append(k)

    if not active_keys:
        return []

    max_len = max(len(buckets[k]) for k in active_keys)
    interleaved: List[Dict[str, Any]] = []
    for i in range(max_len):
        for k in active_keys:
            if i < len(buckets[k]):
                interleaved.append(buckets[k][i])
                if max_total > 0 and len(interleaved) >= max_total:
                    return interleaved
    return interleaved


def bulk_download_clips(
    rows: List[Dict[str, Any]],
    output_dir: Path,
    max_clips: int = 0,
    max_workers: int = 8,
    balanced: bool = True,
    quality: Optional[str] = None,
) -> int:
    """
    Concurrently download missing MP4 clips from metadata rows into challenge-type folders:
    `output_dir/{foul,oob,goaltending,flagrant}/{play_id}.mp4`.
    If `quality='sd'`, prefers `_320x180.mp4` (~4x faster download & CPU decode on Colab).
    """
    output_dir = resolve_project_path(output_dir)
    for ctype in CHALLENGE_TYPES:
        (output_dir / ctype).mkdir(parents=True, exist_ok=True)

    migrate_raw_clips_to_challenge_folders(rows, output_dir)

    ordered = interleave_balanced_rows(rows) if balanced else [upgrade_metadata_row(r) for r in rows]
    pending: List[Tuple[str, str, Path]] = []
    already_cached = 0
    q_low = (quality or "").strip().lower()
    for row in ordered:
        play_id = str(row.get("play_id") or "").strip()
        mp4_url = str(row.get("mp4_url") or "").strip()
        ctype = normalize_challenge_type(row.get("challenge_type"))
        if not play_id or not mp4_url:
            continue
        if q_low == "sd" and "_1280x720.mp4" in mp4_url:
            mp4_url = mp4_url.replace("_1280x720.mp4", "_320x180.mp4")
        elif q_low == "hd" and "_320x180.mp4" in mp4_url:
            mp4_url = mp4_url.replace("_320x180.mp4", "_1280x720.mp4")
        dest = output_dir / ctype / f"{play_id}.mp4"
        if dest.exists() and dest.stat().st_size > 10_240:
            already_cached += 1
            continue
        pending.append((play_id, mp4_url, dest))

    if max_clips > 0:
        need = max(0, max_clips - already_cached)
        if need == 0:
            logger.info("Already have %d cached MP4 clips across challenge folders (target max_clips=%d).", already_cached, max_clips)
            return 0
        pending = pending[:need]

    if not pending:
        logger.info("All %d clips are already downloaded in challenge folders.", already_cached)
        return 0

    logger.info(
        "Downloading %d replay MP4 clips into %s/{foul,oob,goaltending,flagrant}/ (%d workers, cached=%d)...",
        len(pending),
        output_dir.name,
        max_workers,
        already_cached,
    )
    downloaded = 0
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futures = {
            pool.submit(download_mp4_clip, url, dest): pid
            for pid, url, dest in pending
        }
        for fut in as_completed(futures):
            try:
                if fut.result():
                    downloaded += 1
                    if downloaded % 25 == 0 or downloaded == len(pending):
                        logger.info("Download progress: %d / %d clips", downloaded, len(pending))
            except Exception as exc:
                logger.debug("Download worker error for %s: %s", futures[fut], exc)

    logger.info("Download complete: %d newly downloaded (%d total cached).", downloaded, already_cached + downloaded)
    return downloaded


def run_replay_scraper(
    output_dir: Path = BASE_DIR / "data" / "raw",
    metadata_csv: Path = BASE_DIR / "data" / "replay_metadata.csv",
    seasons: Optional[List[str]] = None,
    include_flagrant_reviews: bool = True,
    include_non_foul_challenges: bool = True,
    max_per_class: int = 0,
    max_inspect: int = 0,
    max_clips: int = 0,
    quality: str = "hd",
    workers: int = 12,
    skip_download: bool = False,
) -> List[Dict[str, Any]]:
    """
    End-to-end Stage 1 scraper:
      1. Discovers Coach's Challenge & Flagrant Review replay items across `seasons` via `query_replay_feed`.
      2. Fetches permalink HTML pages concurrently over persistent keep-alive connections.
      3. Classifies both what is being challenged (`foul`, `oob`, `goaltending`, `flagrant`) and the exact
         ruling outcome (`outcome_category` / `call_label` `0..9` + `label` `0..1`).
      4. Saves full metadata to `data/replay_metadata.csv` and downloads MP4 clips into
         `data/raw/{foul,oob,goaltending,flagrant}/`.
    """
    output_dir = resolve_project_path(output_dir)
    metadata_csv = resolve_project_path(metadata_csv)
    target_seasons = seasons or list(DEFAULT_SEASONS)

    triggers = ["coachs-challenge"]
    if include_flagrant_reviews:
        triggers.append("review-of-called-foul")

    existing_rows = load_existing_metadata(metadata_csv)
    seen_play_ids: Set[str] = {str(r["play_id"]) for r in existing_rows}
    seen_permalinks: Set[str] = {
        str(r.get("permalink") or "").strip().rstrip("/")
        for r in existing_rows
        if r.get("permalink")
    }
    rows: List[Dict[str, Any]] = list(existing_rows)

    if rows:
        write_metadata_rows(metadata_csv, rows)
        migrate_raw_clips_to_challenge_folders(rows, output_dir)

    label_counts = Counter(int(r["label"]) for r in rows)
    ctype_counts = Counter(str(r.get("challenge_type", "foul")) for r in rows)
    call_counts = Counter(str(r.get("outcome_category", "foul_stands")) for r in rows)
    logger.info(
        "Loaded %d existing replay rows (challenges=%s) from %s",
        len(rows),
        dict(ctype_counts),
        metadata_csv,
    )

    session = get_thread_session(for_html=True)
    candidate_entries: List[Dict[str, Any]] = []
    seen_feed_keys: Set[str] = set()

    for season in target_seasons:
        for trig in triggers:
            logger.info("Querying NBA Replay Archive API: season=%s, trigger=%s", season, trig)
            items = query_replay_feed(session, season=season, trigger=trig)
            if not items and season == DEFAULT_SEASONS[0] and trig == "coachs-challenge":
                _, fallback_items = fetch_embedded_archive_feed(session)
                items = [
                    it for it in fallback_items
                    if any(
                        k in clean_bracket_list(it.get("trigger")).lower()
                        for k in ("coachs-challenge", "challenge-oob", "challenge-goaltending-bi", "review-of-called-foul")
                    )
                ]

            for it in items:
                title_low = str(it.get("title") or "").lower()
                if trig == "coachs-challenge" and not include_non_foul_challenges:
                    if "challenge of called foul" not in title_low:
                        continue
                pid = make_play_id(it)
                plink = str(it.get("permalink") or "").strip().rstrip("/")
                if pid in seen_play_ids or pid in seen_feed_keys or (plink and plink in seen_permalinks):
                    continue
                seen_feed_keys.add(pid)
                if plink:
                    seen_permalinks.add(plink)
                candidate_entries.append(it)

    # Interleave candidate entries by title prefix so all 4 challenge folders (foul, oob, goaltending, flagrant)
    # are populated evenly right away
    by_prefix: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for it in candidate_entries:
        pref = str(it.get("title") or "").split(":")[0].strip().lower()
        by_prefix[pref].append(it)
    if len(by_prefix) > 1:
        interleaved_candidates: List[Dict[str, Any]] = []
        pref_keys = list(by_prefix.keys())
        max_p_len = max(len(by_prefix[k]) for k in pref_keys)
        for i in range(max_p_len):
            for k in pref_keys:
                if i < len(by_prefix[k]):
                    interleaved_candidates.append(by_prefix[k][i])
        candidate_entries = interleaved_candidates

    if max_inspect > 0 and len(candidate_entries) > max_inspect:
        candidate_entries = candidate_entries[:max_inspect]

    html_workers = min(max(1, workers), 4)
    logger.info(
        "Inspecting %d new replay permalink pages (%d persistent keep-alive workers)...",
        len(candidate_entries),
        html_workers,
    )
    newly_added = 0
    if candidate_entries:
        rss_pages = min(40, max(6, (len(candidate_entries) // 25) + 4))
        logger.info("Pre-harvesting rulings in bulk via WordPress RSS search feeds (%d pages/query)...", rss_pages)
        rss_cache = harvest_rss_ruling_cache(
            session,
            max_pages_per_query=rss_pages,
            workers=html_workers,
            include_flagrant_reviews=include_flagrant_reviews,
            include_non_foul_challenges=include_non_foul_challenges,
        )
        logger.info("Bulk RSS cache harvested %d ruling entries; resolving candidate plays...", len(rss_cache) // 2)
        batch_size = max(32, html_workers * 12)
        with ThreadPoolExecutor(max_workers=html_workers) as pool:
            for b_start in range(0, len(candidate_entries), batch_size):
                if max_per_class > 0 and all(call_counts[c] >= max_per_class for c in CALL_CLASSES):
                    logger.info("Reached target max_per_class=%d across all call classes.", max_per_class)
                    break

                batch = candidate_entries[b_start : b_start + batch_size]
                futures = [
                    pool.submit(
                        fetch_replay_details,
                        entry,
                        quality,
                        include_flagrant_reviews,
                        include_non_foul_challenges,
                        rss_cache,
                    )
                    for entry in batch
                ]
                for fut in as_completed(futures):
                    try:
                        parsed = fut.result()
                    except Exception:
                        parsed = None
                    if parsed is None:
                        continue
                    pid = parsed["play_id"]
                    lbl = int(parsed["label"])
                    cat = str(parsed["outcome_category"])
                    ctype = str(parsed["challenge_type"])
                    if pid in seen_play_ids:
                        continue
                    if max_per_class > 0 and call_counts[cat] >= max_per_class:
                        continue
                    rows.append(parsed)
                    seen_play_ids.add(pid)
                    label_counts[lbl] += 1
                    call_counts[cat] += 1
                    ctype_counts[ctype] += 1
                    newly_added += 1

                write_metadata_rows(metadata_csv, rows)
                logger.info(
                    "Parsed %d/%d pages | Catalog total: %d (types=%s)",
                    min(b_start + batch_size, len(candidate_entries)),
                    len(candidate_entries),
                    len(rows),
                    dict(ctype_counts),
                )

    write_metadata_rows(metadata_csv, rows)
    outcome_counts = Counter(r.get("outcome_category", "unknown") for r in rows)
    logger.info(
        "Metadata sync complete: +%d new rows -> %d total | challenge_folders=%s | call_outcomes=%s",
        newly_added,
        len(rows),
        dict(ctype_counts),
        dict(outcome_counts),
    )

    if not skip_download and rows:
        bulk_download_clips(
            rows=rows,
            output_dir=output_dir,
            max_clips=max_clips,
            max_workers=min(max(1, workers), 12),
            balanced=True,
        )

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1: Scrape official.nba.com/replay/archive/ into challenge-type folders (foul, oob, goaltending, flagrant)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "data" / "raw",
        help="Directory for downloaded raw replay MP4s organized by challenge type (default: data/raw)",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=BASE_DIR / "data" / "replay_metadata.csv",
        help="Path to replay metadata CSV (default: data/replay_metadata.csv)",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=list(DEFAULT_SEASONS),
        help="NBA seasons to query (default: all seasons 2026-27 .. 2019-20)",
    )
    parser.add_argument(
        "--include-flagrant-reviews",
        action="store_true",
        default=True,
        help="Include Flagrant / Called-Foul reviews in data/raw/flagrant/ (enabled by default)",
    )
    parser.add_argument(
        "--no-flagrant-reviews",
        action="store_true",
        help="Exclude Flagrant reviews and only scrape Coach's Challenges",
    )
    parser.add_argument(
        "--include-non-foul-challenges",
        action="store_true",
        default=True,
        help="Include OOB & Goaltending/BI challenges in data/raw/{oob,goaltending}/ (enabled by default)",
    )
    parser.add_argument(
        "--foul-challenges-only",
        action="store_true",
        help="Restrict scraping strictly to Challenge of Called Foul plays",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Optional cap on metadata rows collected per call class (0 = unlimited)",
    )
    parser.add_argument(
        "--max-inspect",
        type=int,
        default=0,
        help="Max new archive permalink pages to inspect per run (0 = inspect all discovered; default: 0)",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=0,
        help="Max total MP4 clips to have downloaded in data/raw/ (0 = download all; default: 0)",
    )
    parser.add_argument(
        "--quality",
        choices=("hd", "sd"),
        default="hd",
        help="Video resolution to download: 'hd' (1280x720) or 'sd' (320x180) (default: hd)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="Concurrent worker threads (default: 12)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only build/update data/replay_metadata.csv without downloading MP4 clips",
    )
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Skip web scraping and download missing MP4 clips from existing replay_metadata.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    inc_non_foul = False if cli_args.foul_challenges_only else True
    inc_flagrant = False if cli_args.no_flagrant_reviews else True
    if cli_args.download_missing:
        existing = load_existing_metadata(resolve_project_path(cli_args.metadata_csv))
        bulk_download_clips(
            rows=existing,
            output_dir=cli_args.output_dir,
            max_clips=cli_args.max_clips,
            max_workers=cli_args.workers,
            balanced=True,
        )
    else:
        run_replay_scraper(
            output_dir=cli_args.output_dir,
            metadata_csv=cli_args.metadata_csv,
            seasons=cli_args.seasons,
            include_flagrant_reviews=inc_flagrant,
            include_non_foul_challenges=inc_non_foul,
            max_per_class=cli_args.max_per_class,
            max_inspect=cli_args.max_inspect,
            max_clips=cli_args.max_clips,
            quality=cli_args.quality,
            workers=cli_args.workers,
            skip_download=cli_args.skip_download,
        )
