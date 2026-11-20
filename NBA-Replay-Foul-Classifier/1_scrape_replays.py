#!/usr/bin/env python3
"""
STAGE 1: Official NBA Replay Archive Scraper & Binary Dataset Builder (`1_scrape_replays.py`)

Scrapes official NBA Replay Center clips from:
  - Archive Portal : https://official.nba.com/replay/archive/
  - Replay Feed API: https://official.nba.com/wp-json/api/v1/query_replay_feed
  - Replay CDN     : https://videos.nba.com/nba/official-replay/media/...

Builds a clean binary classification dataset (`foul` vs. `no_foul`) by parsing the verbatim
Replay Center ruling on each archive permalink page:
  - Class 1 (`foul`):
      • `Challenge of Called Foul` where the called foul stood / was upheld / confirmed
      • `Challenge of Called Foul` overturned to ANOTHER foul (e.g. defensive -> offensive foul)
      • Optional: `Review of Called Foul` (flagrant reviews), clear-path / off-ball fouls,
        or OOB/goaltending challenges where a proximate foul was assessed
  - Class 0 (`no_foul`):
      • `Challenge of Called Foul` explicitly overturned to NO FOUL (hard negatives with real contact)
      • Optional (`--include-non-foul-challenges`): clean `Challenge OOB` / `Challenge Goaltending/BI`
        plays where no proximate foul was assessed

Outputs:
  - `data/replay_metadata.csv`
  - `data/raw/foul/{play_id}.mp4`
  - `data/raw/no_foul/{play_id}.mp4`
"""

from __future__ import annotations

import argparse
import csv
import html as html_lib
import json
import logging
import re
import threading
import time
import urllib.parse
import urllib.request
from collections import Counter
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
    "title",
    "teams",
    "outcome_category",
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


_THREAD_LOCAL = threading.local()


def build_session(retries: int = 3, backoff_factor: float = 0.8) -> Any:
    """Create a resilient HTTP session with retry backoff (or stdlib urllib fallback)."""
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
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_thread_session() -> Any:
    sess = getattr(_THREAD_LOCAL, "session", None)
    if sess is None:
        sess = build_session()
        _THREAD_LOCAL.session = sess
    return sess


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


def query_replay_feed(
    session: Any,
    season: str,
    trigger: str = "coachs-challenge",
    max_items: int = 0,
    sleep_sec: float = 0.25,
) -> List[Dict[str, Any]]:
    """
    Fetch replay archive feed items for a `(season, trigger)` slice from
    `/wp-json/api/v1/query_replay_feed`, handling pagination safely.
    """
    collected: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    offset = 0
    page_guard = 0

    while page_guard < 10:
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

    # Deduplicate while preserving order
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
        r"overturned\s+(?:and\s+(?:changed|ruled)\s+)?to\s+(?:a|an)\s+"
        r"(?:[\w\-]+\s+){0,3}foul\b",
        re.I,
    ),
    re.compile(
        r"(?:changed|modified|upgraded|downgraded|reclassified|assessed)\s+to\s+(?:a|an)\s+"
        r"(?:[\w\-]+\s+){0,3}foul\b",
        re.I,
    ),
    re.compile(
        r"overturned[^.]{0,80}\b(?:defensive|offensive|personal|shooting|loose[\s\-]ball|"
        r"blocking|common|flagrant|away[\s\-]from[\s\-]play|take|transition|clear[\s\-]path)\s+foul\b",
        re.I,
    ),
    re.compile(
        r"\bproximate\s+foul\b|\bfoul\s+(?:was\s+)?(?:called|assessed|committed)\s+on\b",
        re.I,
    ),
)

_OVERTURNED_TO_NO_FOUL_PATTERNS = (
    re.compile(r"\boverturned\s+to\s+(?:a\s+)?no[\s\-]?(?:foul|call)\b", re.I),
    re.compile(r"\bno\s+foul\s+(?:occurred|was\s+committed|on\s+the\s+play)\b", re.I),
    re.compile(r"\b(?:ruled|determined)\s+(?:there\s+was\s+)?no\s+foul\b", re.I),
    re.compile(r"\boverturned\s+(?:and\s+a\s+)?jump\s+ball\b", re.I),
    re.compile(r"\boverturned\s+to\s+(?:an?\s+)?out[\s\-]of[\s\-]bounds\b", re.I),
    re.compile(r"\boverturned\b[^.]{0,60}\b(?:legal\s+guarding|incidental\s+contact|marginal\s+contact|clean\s+block|no\s+illegal\s+contact)\b", re.I),
    re.compile(r"\bchallenge\s+(?:was\s+)?successful\b[^.]{0,80}\bno\s+foul\b", re.I),
    re.compile(r"\bfoul\b[^.]{0,60}\b(?:was\s+)?rescinded\b", re.I),
)

_FOUL_UPHELD_PATTERNS = (
    re.compile(r"\bwould\s+stand\b|\bcall\s+(?:on\s+the\s+floor\s+)?stands\b|\bruling\s+(?:on\s+the\s+floor\s+)?stands\b", re.I),
    re.compile(r"\b(?:was|is)\s+(?:upheld|confirmed)\b", re.I),
    re.compile(r"\bchallenge\s+(?:was|is)\s+unsuccessful\b", re.I),
    re.compile(r"\bremains\s+a\s+(?:[\w\-]+\s+)?foul\b", re.I),
    re.compile(r"\bfoul\s+(?:was\s+)?(?:upheld|confirmed|stands|stood)\b", re.I),
)


def classify_replay_ruling(
    title: str,
    trigger: str,
    ruling_text: str,
    include_flagrant_reviews: bool = False,
    include_non_foul_challenges: bool = False,
) -> Optional[Tuple[int, str]]:
    """
    Map an NBA Replay Archive entry + verbatim ruling text to a strict binary label:
      - `(1, outcome_category)` -> **FOUL** occurred on the play
      - `(0, outcome_category)` -> **NO FOUL** occurred on the play
      - `None`                  -> Excluded or ambiguous entry (skipped to protect label purity)
    """
    title_low = (title or "").strip().lower()
    trigger_low = clean_bracket_list(trigger).lower()
    text_low = re.sub(r"\s+", " ", (ruling_text or "").strip()).lower()

    if not text_low:
        return None

    # 1. Coach's Challenge of Called Foul (primary source of BOTH hard-negative no_foul and foul!)
    is_foul_challenge = (
        "challenge of called foul" in title_low
        or "foul challenge" in text_low
        or (
            "coachs-challenge" in trigger_low
            and "out of bounds" not in text_low
            and "challenge oob" not in title_low
            and "goaltending" not in title_low
            and "goaltending" not in text_low
            and "basket interference" not in text_low
            and "foul" in text_low
        )
    )
    if is_foul_challenge:
        # Check explicit "overturned to NO FOUL" first (must NOT also say overturned to another foul)
        has_other_foul = any(p.search(text_low) for p in _OVERTURNED_TO_OTHER_FOUL_PATTERNS)
        has_no_foul = any(p.search(text_low) for p in _OVERTURNED_TO_NO_FOUL_PATTERNS)
        has_upheld = any(p.search(text_low) for p in _FOUL_UPHELD_PATTERNS)

        if has_other_foul and not has_no_foul:
            return 1, "overturned_to_other_foul"
        if has_no_foul and not has_other_foul:
            return 0, "overturned_to_no_foul"
        if has_upheld and "overturned" not in text_low:
            return 1, "foul_upheld"

        # Older Replay Center phrasing: "the foul challenged by X was overturned." (no other foul mentioned)
        if "overturned" in text_low and not has_other_foul:
            if not re.search(r"overturned[^.]{0,60}\bfoul\s+on\b", text_low):
                return 0, "overturned_to_no_foul"
        return None

    # 2. Optional: Non-foul Coach's Challenges (OOB / Goaltending / BI)
    is_oob_or_gt_challenge = (
        "challenge oob" in title_low
        or "challenge-oob" in trigger_low
        or "out of bounds challenge" in text_low
        or "challenge goaltending" in title_low
        or "challenge-goaltending-bi" in trigger_low
        or "goaltending" in title_low
    )
    if is_oob_or_gt_challenge:
        if not include_non_foul_challenges:
            return None
        # Even on OOB challenges, NBA rules allow calling a proximate foul on review!
        if any(p.search(text_low) for p in _OVERTURNED_TO_OTHER_FOUL_PATTERNS) or re.search(
            r"\b(?:foul\s+on|offensive\s+foul|defensive\s+foul|personal\s+foul|loose[\s\-]ball\s+foul)\b",
            text_low,
        ):
            return 1, "proximate_foul_assessed"
        return 0, "non_foul_challenge_clean"

    # 3. Optional: Flagrant / Clear-Path / Off-Ball Called Foul Reviews
    is_called_foul_review = (
        "review of called foul" in title_low
        or "review-of-called-foul" in trigger_low
        or "clear-path-to-the" in title_low
        or "clear-path-to-the-basket-foul" in trigger_low
        or "off-ball foul" in title_low
        or "off-ball-foul-timing" in trigger_low
    )
    if is_called_foul_review:
        if not include_flagrant_reviews:
            return None
        if any(p.search(text_low) for p in _OVERTURNED_TO_NO_FOUL_PATTERNS):
            return 0, "review_rescinded_no_foul"
        # "did not meet the criteria for a flagrant foul" -> common foul still stands!
        if any(
            kw in text_low
            for kw in (
                "flagrant",
                "common foul",
                "clear path",
                "transition take foul",
                "off-ball foul",
                "personal foul",
                "remains a foul",
                "foul called",
            )
        ):
            return 1, "called_foul_review"
        return None

    return None


def fetch_replay_details(
    entry: Dict[str, Any],
    quality: str = "hd",
    include_flagrant_reviews: bool = False,
    include_non_foul_challenges: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Fetch an archive entry's permalink HTML, parse ruling text + MP4 URL,
    and return a normalized metadata dict if it cleanly maps to `foul` (1) or `no_foul` (0).
    """
    permalink = str(entry.get("permalink") or "").strip()
    if not permalink:
        return None

    session = get_thread_session()
    try:
        resp = session.get(permalink, headers=OFFICIAL_HEADERS, timeout=20)
        resp.raise_for_status()
        ruling_text, mp4_url = parse_permalink_html(resp.text, quality=quality)
    except Exception as exc:
        logger.debug("Failed fetching permalink %s: %s", permalink, exc)
        return None

    title = html_lib.unescape(str(entry.get("title") or "").strip())
    trigger = clean_bracket_list(entry.get("trigger"))
    classified = classify_replay_ruling(
        title=title,
        trigger=trigger,
        ruling_text=ruling_text,
        include_flagrant_reviews=include_flagrant_reviews,
        include_non_foul_challenges=include_non_foul_challenges,
    )
    if classified is None:
        return None

    label, outcome_category = classified
    if not mp4_url:
        mp4_url = derive_mp4_from_thumbnail(str(entry.get("thumbnail_url") or ""), quality=quality)
    if not mp4_url:
        return None

    play_id = make_play_id(entry)
    game_id = normalize_game_id(entry.get("game_id"))
    class_dir = "foul" if label == 1 else "no_foul"
    rel_video_path = (Path("data") / "raw" / class_dir / f"{play_id}.mp4").as_posix()

    return {
        "play_id": play_id,
        "archive_id": str(entry.get("id") or ""),
        "ops_video_id": str(entry.get("opsreporting_video_id") or ""),
        "game_id": game_id,
        "season": str(entry.get("_queried_season") or entry.get("season") or ""),
        "game_date": str(entry.get("game_date") or entry.get("date") or ""),
        "trigger": trigger,
        "title": title,
        "teams": clean_bracket_list(entry.get("teams")),
        "outcome_category": outcome_category,
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
    """Stream-download a single MP4 replay clip safely using atomic `.part` rename."""
    if dest_path.exists() and dest_path.stat().st_size > 10_240:
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(".mp4.part")
    session = get_thread_session()
    try:
        with session.get(mp4_url, headers=OFFICIAL_HEADERS, stream=True, timeout=30) as resp:
            resp.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
        if tmp_path.exists() and tmp_path.stat().st_size > 10_240:
            tmp_path.replace(dest_path)
            return True
        tmp_path.unlink(missing_ok=True)
        return False
    except Exception as exc:
        logger.debug("Failed downloading %s: %s", mp4_url, exc)
        tmp_path.unlink(missing_ok=True)
        return False


def load_existing_metadata(metadata_csv: Path) -> List[Dict[str, Any]]:
    if not metadata_csv.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(metadata_csv, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("play_id"):
                rows.append(dict(row))
    return rows


def write_metadata_rows(metadata_csv: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows and metadata_csv.exists() and metadata_csv.stat().st_size > 100:
        return
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = metadata_csv.with_suffix(".csv.tmp")
    with open(tmp_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp_csv.replace(metadata_csv)


def interleave_balanced_rows(rows: List[Dict[str, Any]], max_total: int = 0) -> List[Dict[str, Any]]:
    """Interleave `foul` (1) and `no_foul` (0) rows 1:1 so any prefix is balanced."""
    foul_rows = [r for r in rows if int(r.get("label", 0)) == 1]
    no_foul_rows = [r for r in rows if int(r.get("label", 0)) == 0]
    interleaved: List[Dict[str, Any]] = []
    for i in range(max(len(foul_rows), len(no_foul_rows))):
        if i < len(no_foul_rows):
            interleaved.append(no_foul_rows[i])
        if i < len(foul_rows):
            interleaved.append(foul_rows[i])
    if max_total > 0:
        return interleaved[:max_total]
    return interleaved


def bulk_download_clips(
    rows: List[Dict[str, Any]],
    output_dir: Path,
    max_clips: int = 0,
    max_workers: int = 6,
    balanced: bool = True,
) -> int:
    """Concurrently download missing MP4 clips from metadata rows into `output_dir/{foul,no_foul}/`."""
    output_dir = resolve_project_path(output_dir)
    (output_dir / "foul").mkdir(parents=True, exist_ok=True)
    (output_dir / "no_foul").mkdir(parents=True, exist_ok=True)

    ordered = interleave_balanced_rows(rows) if balanced else list(rows)
    pending: List[Tuple[str, str, Path]] = []
    already_cached = 0
    for row in ordered:
        play_id = str(row.get("play_id") or "").strip()
        mp4_url = str(row.get("mp4_url") or "").strip()
        label = int(row.get("label", 0))
        if not play_id or not mp4_url:
            continue
        dest = output_dir / ("foul" if label == 1 else "no_foul") / f"{play_id}.mp4"
        if dest.exists() and dest.stat().st_size > 10_240:
            already_cached += 1
            continue
        pending.append((play_id, mp4_url, dest))

    if max_clips > 0:
        need = max(0, max_clips - already_cached)
        if need == 0:
            logger.info("Already have %d cached MP4 clips (target max_clips=%d).", already_cached, max_clips)
            return 0
        pending = pending[:need]

    if not pending:
        logger.info("All %d clips are already downloaded.", already_cached)
        return 0

    logger.info(
        "Downloading %d replay MP4 clips with %d workers (already cached: %d)...",
        len(pending),
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
                    if downloaded % 20 == 0 or downloaded == len(pending):
                        logger.info("Download progress: %d / %d clips", downloaded, len(pending))
            except Exception as exc:
                logger.debug("Download worker error for %s: %s", futures[fut], exc)

    logger.info("Download complete: %d newly downloaded (%d total cached).", downloaded, already_cached + downloaded)
    return downloaded


def run_replay_scraper(
    output_dir: Path = BASE_DIR / "data" / "raw",
    metadata_csv: Path = BASE_DIR / "data" / "replay_metadata.csv",
    seasons: Optional[List[str]] = None,
    include_flagrant_reviews: bool = False,
    include_non_foul_challenges: bool = False,
    max_per_class: int = 0,
    max_inspect: int = 400,
    max_clips: int = 120,
    quality: str = "hd",
    workers: int = 8,
    skip_download: bool = False,
) -> List[Dict[str, Any]]:
    """
    End-to-end Stage 1 scraper:
      1. Discovers replay items across `seasons` via `query_replay_feed`.
      2. Fetches permalink HTML pages concurrently to parse verbatim rulings & MP4 URLs.
      3. Saves binary-labeled metadata (`1 = foul`, `0 = no_foul`) to `data/replay_metadata.csv`.
      4. Downloads balanced MP4 clips into `data/raw/{foul,no_foul}/`.
    """
    output_dir = resolve_project_path(output_dir)
    metadata_csv = resolve_project_path(metadata_csv)
    target_seasons = seasons or list(DEFAULT_SEASONS[:3])

    triggers = ["coachs-challenge"]
    if include_flagrant_reviews:
        triggers.extend(["review-of-called-foul", "clear-path-to-the-basket-foul", "off-ball-foul-timing"])

    existing_rows = load_existing_metadata(metadata_csv)
    seen_play_ids: Set[str] = {str(r["play_id"]) for r in existing_rows}
    rows: List[Dict[str, Any]] = list(existing_rows)

    label_counts = Counter(int(r["label"]) for r in rows)
    logger.info(
        "Loaded %d existing replay rows (foul=%d, no_foul=%d) from %s",
        len(rows),
        label_counts[1],
        label_counts[0],
        metadata_csv,
    )

    session = build_session()
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
                    if "coachs-challenge" in clean_bracket_list(it.get("trigger")).lower()
                ]

            for it in items:
                title_low = str(it.get("title") or "").lower()
                # Fast pre-filter before fetching permalink HTML
                if trig == "coachs-challenge" and not include_non_foul_challenges:
                    if "challenge of called foul" not in title_low:
                        continue
                pid = make_play_id(it)
                if pid in seen_play_ids or pid in seen_feed_keys:
                    continue
                seen_feed_keys.add(pid)
                candidate_entries.append(it)

    if max_inspect > 0 and len(candidate_entries) > max_inspect:
        candidate_entries = candidate_entries[:max_inspect]

    logger.info("Inspecting %d new replay permalink pages (%d workers)...", len(candidate_entries), workers)
    newly_added = 0
    if candidate_entries:
        batch_size = max(16, workers * 4)
        for b_start in range(0, len(candidate_entries), batch_size):
            if max_per_class > 0 and label_counts[0] >= max_per_class and label_counts[1] >= max_per_class:
                logger.info("Reached target max_per_class=%d for both foul and no_foul.", max_per_class)
                break

            batch = candidate_entries[b_start : b_start + batch_size]
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                futures = [
                    pool.submit(
                        fetch_replay_details,
                        entry,
                        quality,
                        include_flagrant_reviews,
                        include_non_foul_challenges,
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
                    if pid in seen_play_ids:
                        continue
                    if max_per_class > 0 and label_counts[lbl] >= max_per_class:
                        continue
                    rows.append(parsed)
                    seen_play_ids.add(pid)
                    label_counts[lbl] += 1
                    newly_added += 1

            write_metadata_rows(metadata_csv, rows)
            logger.info(
                "Parsed %d/%d pages | Catalog total: %d (foul=%d, no_foul=%d)",
                min(b_start + batch_size, len(candidate_entries)),
                len(candidate_entries),
                len(rows),
                label_counts[1],
                label_counts[0],
            )

    write_metadata_rows(metadata_csv, rows)
    outcome_counts = Counter(r.get("outcome_category", "unknown") for r in rows)
    logger.info(
        "Metadata sync complete: +%d new rows -> %d total | foul=%d, no_foul=%d | outcomes=%s",
        newly_added,
        len(rows),
        label_counts[1],
        label_counts[0],
        dict(outcome_counts),
    )

    if not skip_download and rows:
        bulk_download_clips(
            rows=rows,
            output_dir=output_dir,
            max_clips=max_clips,
            max_workers=min(workers, 8),
            balanced=True,
        )

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1: Scrape official.nba.com/replay/archive/ into a binary Foul vs. No Foul dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "data" / "raw",
        help="Directory for downloaded raw replay MP4s (default: data/raw)",
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
        default=list(DEFAULT_SEASONS[:3]),
        help="NBA seasons to query (default: 2026-27 2025-26 2024-25)",
    )
    parser.add_argument(
        "--include-flagrant-reviews",
        action="store_true",
        help="Also include Flagrant / Clear-Path / Off-Ball called-foul reviews as Class 1 (foul)",
    )
    parser.add_argument(
        "--include-non-foul-challenges",
        action="store_true",
        help="Also include OOB & Goaltending/BI challenges (clean -> no_foul, proximate foul -> foul)",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Optional cap on metadata rows collected per class (0 = unlimited)",
    )
    parser.add_argument(
        "--max-inspect",
        type=int,
        default=400,
        help="Max new archive permalink pages to inspect per run (0 = all discovered; default: 400)",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=120,
        help="Max total MP4 clips to have downloaded in data/raw/ (0 = download all; default: 120)",
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
        default=8,
        help="Concurrent worker threads for HTML scraping & downloads (default: 8)",
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
            include_flagrant_reviews=cli_args.include_flagrant_reviews,
            include_non_foul_challenges=cli_args.include_non_foul_challenges,
            max_per_class=cli_args.max_per_class,
            max_inspect=cli_args.max_inspect,
            max_clips=cli_args.max_clips,
            quality=cli_args.quality,
            workers=cli_args.workers,
            skip_download=cli_args.skip_download,
        )
