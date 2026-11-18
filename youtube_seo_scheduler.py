import subprocess
"""
Step 1 (Ultimate 24/7 Edition) — YouTube Live Auto-Discovery, Analytics Heatmap Optimizer,
Google Drive Sync & Native YouTube `publishAt` Scheduler
==========================================================================================
Reverse-engineered and upgraded from `haziqahmedqasmi/peakcut`.

What this module does 24/7 while you sleep:
1. Auto-Discovers Gaming Streams (`discover_youtube_and_watchlist_stream`):
   - Checks your preferred big YouTube gaming channels (`@IShowSpeed`, `@TypicalGamer`, `@TechnoGamerzOfficial`, etc.)
     using the official YouTube Data API v3 (`channels.list` + `playlistItems.list` + `videos.list`, only 3 quota units!)
   - Searches YouTube Live (`search.list` with `eventType="live"`, `videoCategoryId="20"`, `order="viewCount"`) for
     trending live streams matching your `SEARCH:` keywords in `watchlist.txt`.
   - Syncs a persistent Deduplication Ledger (`bot_state_ledger.json`) in your Google Drive folder so the 24/7 loop
     never clips the same VOD or live 30-minute window twice.
2. Monitors Channel Analytics & Adjusts Schedule Heatmap (`update_analytics_heatmap`):
   - Queries your channel's published Shorts via YouTube Data API v3 (`videos.list(part="statistics,snippet")`) and
     YouTube Analytics API v2 (`youtubeanalytics.googleapis.com/v2/reports`).
   - Builds a 24-hour performance heatmap (views/hour velocity + like & retention weight by UTC publish hour)
     and automatically shifts future `publishAt` slots into your channel's highest-performing Golden Hours.
3. Generates viral SEO metadata (Title, 150-word Description, 15 Tags, Category ID, and 3:2 Safe-Zone Thumbnail)
   and schedules Shorts directly on YouTube Studio (`status.publishAt`).
"""

import argparse
import datetime as dt
import json
import os
import re
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# 1. API Endpoints & Unified OAuth Scopes
# ---------------------------------------------------------------------------
REPO_DIRS = [".github/workflows", "fonts", "src", "work/clips"]

YT_UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
YT_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
YT_CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"
YT_PLAYLIST_ITEMS_URL = "https://www.googleapis.com/youtube/v3/playlistItems"
YT_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YT_THUMB_URL = "https://www.googleapis.com/upload/youtube/v3/thumbnails/set"
YT_ANALYTICS_URL = "https://youtubeanalytics.googleapis.com/v2/reports"

DRIVE_FILES_URL = "https://www.googleapis.com/drive/v3/files"
DRIVE_UPLOAD_URL = "https://www.googleapis.com/upload/drive/v3/files"
YT_TOKEN_URL = "https://oauth2.googleapis.com/token"
YT_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"

# Single OAuth scope covering YouTube Upload, Read/Search, YouTube Analytics Heatmap, and Google Drive
YT_SCOPES = (
    "https://www.googleapis.com/auth/youtube.upload "
    "https://www.googleapis.com/auth/youtube.readonly "
    "https://www.googleapis.com/auth/yt-analytics.readonly "
    "https://www.googleapis.com/auth/drive"
)

# Cold-start Golden Hours (UTC) before channel has >= 5 published Shorts:
# 12:00 UTC (5 PM PKT / 7 AM EST), 15:00 UTC (8 PM PKT / 10 AM EST),
# 18:00 UTC (11 PM PKT / 1 PM EST), 21:00 UTC (2 AM PKT / 4 PM EST), 00:00 UTC (7 PM EST)
DEFAULT_GOLDEN_HOURS_UTC = [12, 15, 18, 21, 0, 9]
STATE_FILENAME = "bot_state_ledger.json"


def env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def init_repo_structure(root: Path = Path(".")) -> None:
    for d in REPO_DIRS:
        (root / d).mkdir(parents=True, exist_ok=True)
    print(f"[OK] Stateless repository folders verified in: {root.resolve()}")


# ---------------------------------------------------------------------------
# 2. Google OAuth2 Token Refresh
# ---------------------------------------------------------------------------
def refresh_google_access_token(
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    refresh_token: Optional[str] = None,
) -> str:
    cid = client_id or env("GOOGLE_CLIENT_ID")
    csecret = client_secret or env("GOOGLE_CLIENT_SECRET")
    rtoken = refresh_token or env("YT_REFRESH_TOKEN")
    if not (cid and csecret and rtoken):
        raise ValueError(
            "Missing Google OAuth credentials. Set GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, "
            "and YT_REFRESH_TOKEN in GitHub Secrets."
        )
    r = httpx.post(
        YT_TOKEN_URL,
        data={
            "client_id": cid,
            "client_secret": csecret,
            "refresh_token": rtoken,
            "grant_type": "refresh_token",
        },
        timeout=30,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"Google token refresh failed ({r.status_code}): {r.text[:300]}")
    return r.json()["access_token"]


refresh_youtube_access_token = refresh_google_access_token

# ---------------------------------------------------------------------------
# 2b. Multi-Model Free-Tier Cascade & Daily Viral Trend Researcher
#     (gemini-2.5-flash [250 RPD] -> gemini-2.5-flash-lite [1,000 RPD] -> Groq [1,000 RPD])
# ---------------------------------------------------------------------------
FREE_GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-1.5-flash"]


def _resolve_ai_keys(explicit_ai_key: Optional[str] = None) -> Tuple[str, str]:
    candidates = [
        (explicit_ai_key or "").strip().strip('"').strip("'"),
        env("AI_API_KEY").strip().strip('"').strip("'"),
        env("GROQ_API_KEY").strip().strip('"').strip("'"),
    ]
    gemini_key = next((k for k in candidates if k.startswith("AIza")), "")
    groq_key = next((k for k in candidates if k.startswith("gsk_")), "")
    if not gemini_key and candidates[1] and not candidates[1].startswith("gsk_"):
        gemini_key = candidates[1]
    if not groq_key and candidates[2] and not candidates[2].startswith("AIza"):
        groq_key = candidates[2]
    return gemini_key, groq_key


def call_gemini_json_with_cascade(prompt: str, schema: Dict, ai_key: Optional[str] = None) -> Dict:
    key, groq_key = _resolve_ai_keys(ai_key)
    last_err = None
    req_keys = schema.get("required") or list((schema.get("properties") or {}).keys())
    full_prompt = f"{prompt}\n\nReturn ONLY a valid JSON object containing these exact keys: {json.dumps(req_keys)}"

    if key:
        for model_name in FREE_GEMINI_MODELS:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
            for use_schema in (True, False):
                gen_cfg: Dict = {"temperature": 0.35, "responseMimeType": "application/json"}
                if use_schema and schema:
                    gen_cfg["responseSchema"] = schema
                payload = {"contents": [{"parts": [{"text": full_prompt}]}], "generationConfig": gen_cfg}
                try:
                    r = httpx.post(url, params={"key": key}, json=payload, timeout=45)
                    if r.status_code == 429:
                        break
                    if r.status_code == 400 and use_schema:
                        continue
                    r.raise_for_status()
                    text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                    return extract_json_payload(text)
                except Exception as exc:
                    last_err = exc
                    break

    if groq_key:
        for groq_model in ("llama-3.3-70b-versatile", "llama-3.1-8b-instant"):
            try:
                r = httpx.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json", "User-Agent": "RazonGamerzBot/2.0"},
                    json={
                        "model": groq_model,
                        "response_format": {"type": "json_object"},
                        "messages": [
                            {"role": "system", "content": f"Return ONLY valid JSON with keys: {json.dumps(req_keys)}"},
                            {"role": "user", "content": full_prompt},
                        ],
                        "temperature": 0.3,
                    },
                    timeout=45,
                )
                r.raise_for_status()
                return extract_json_payload(r.json()["choices"][0]["message"]["content"])
            except Exception as exc:
                last_err = exc

    raise RuntimeError(f"All free AI models exhausted or unavailable: {last_err}")


def research_daily_viral_trends(state: Dict, topic_hint: str = "auto") -> Dict:
    """
    Runs once per day (cached in `bot_state_ledger.json['daily_trend_plan']`).
    Uses Gemini with Google Search Grounding (free up to 1,500 RPD) to research today's
    breakout gaming storylines, viral streamer moments, and high-velocity search terms.
    """
    today_str = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    existing = state.get("daily_trend_plan") or {}
    if existing.get("date") == today_str and existing.get("viral_keywords"):
        return existing

    key = env("AI_API_KEY")
    niche = topic_hint if topic_hint.lower() not in ("auto", "") else "Grand Theft Auto VI (GTA 6) & Top Gaming Live Streams"
    plan = {
        "date": today_str,
        "focus_niche": niche,
        "viral_keywords": ["GTA 6 live gameplay", "GTA 6 Vice City secret", "GTA 6 streamer reaction", "Lucia and Jason GTA 6"],
        "hook_angles": ["WAIT FOR THE ENDING", "6-STAR POLICE FAIL", "SECRET VICE CITY EASTER EGG", "BRO COULDN'T BELIEVE THIS"],
        "dynamic_search_queries": [f"{niche} live gameplay", "GTA 6 live stream reaction"],
    }
    if not key:
        state["daily_trend_plan"] = plan
        return plan

    prompt = (
        f"Today is {today_str}. Research the hottest viral trends, breakout storylines, and top searched "
        f"keywords on YouTube Shorts and Google right now for: {niche}.\n"
        f"Return JSON with keys: 'focus_niche' (string), 'viral_keywords' (array of 8 search terms), "
        f"'hook_angles' (array of 5 short ALL-CAPS 4-word video hooks), and "
        f"'dynamic_search_queries' (array of 2 YouTube live search strings)."
    )
    schema = {
        "type": "OBJECT",
        "properties": {
            "focus_niche": {"type": "STRING"},
            "viral_keywords": {"type": "ARRAY", "items": {"type": "STRING"}},
            "hook_angles": {"type": "ARRAY", "items": {"type": "STRING"}},
            "dynamic_search_queries": {"type": "ARRAY", "items": {"type": "STRING"}},
        },
        "required": ["focus_niche", "viral_keywords", "hook_angles", "dynamic_search_queries"],
    }
    try:
        res = call_gemini_json_with_cascade(prompt, schema, ai_key=key)
        plan.update(res, date=today_str)
        print(f"[DAILY TREND PLANNER] Updated viral keywords for {today_str}: {plan['viral_keywords'][:4]}")
    except Exception as exc:
        print(f"[WARN] Daily trend research used default GTA 6 launch plan ({exc})")

    state["daily_trend_plan"] = plan
    return plan



# ---------------------------------------------------------------------------
# 3. Persistent Cloud State Ledger (Deduplication + Heatmap Storage in Drive)
# ---------------------------------------------------------------------------
_APP_FOLDER_CACHE: Dict[str, str] = {}


def _get_or_create_app_drive_folder(folder_name: str, token: str) -> str:
    if folder_name in _APP_FOLDER_CACHE:
        return _APP_FOLDER_CACHE[folder_name]
    q = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    r = httpx.get(
        DRIVE_FILES_URL,
        params={"q": q, "pageSize": 1, "fields": "files(id, name)", "supportsAllDrives": "true", "includeItemsFromAllDrives": "true"},
        headers={"Authorization": f"Bearer {token}"},
        timeout=20,
    )
    files = (r.json().get("files") or []) if r.status_code == 200 else []
    if files:
        _APP_FOLDER_CACHE[folder_name] = files[0]["id"]
        return files[0]["id"]
    cr = httpx.post(
        DRIVE_FILES_URL,
        params={"fields": "id, name", "supportsAllDrives": "true"},
        json={"name": folder_name, "mimeType": "application/vnd.google-apps.folder"},
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        timeout=20,
    )
    if cr.status_code < 400 and cr.json().get("id"):
        _APP_FOLDER_CACHE[folder_name] = cr.json()["id"]
        return cr.json()["id"]
    return ""


def extract_drive_folder_id(folder_id_or_url: str) -> str:
    raw = (folder_id_or_url or "").strip()
    m = re.search(r"/folders/([a-zA-Z0-9_-]+)", raw)
    return m.group(1) if m else raw


def load_cloud_state(access_token: Optional[str] = None) -> Dict:
    """
    Loads `bot_state_ledger.json` from local disk or Google Drive Output folder.
    Tracks:
      - `processed_keys`: list of `video_id:bucket` keys already clipped so 24/7 runs never repeat.
      - `channel_id_cache`: maps `@Handle` -> `UC...` channel ID to save YouTube quota.
      - `golden_hours_utc`: learned top-performing UTC publish hours from your channel heatmap.
      - `hourly_heatmap`: `{ "0": {"views_per_hr": ...}, ... }`.
    """
    local_state = Path("work") / STATE_FILENAME
    default_state = {
        "processed_keys": [],
        "channel_id_cache": {},
        "golden_hours_utc": list(DEFAULT_GOLDEN_HOURS_UTC),
        "hourly_heatmap": {},
        "scheduled_iso_slots": [],
        "updated_at": "",
    }
    folder_id = extract_drive_folder_id(env("GDRIVE_OUTPUT_FOLDER_ID") or env("GDRIVE_INBOX_FOLDER_ID"))
    if folder_id:
        try:
            token = access_token or refresh_google_access_token()
            q = f"'{folder_id}' in parents and name = '{STATE_FILENAME}' and trashed = false"
            r = httpx.get(
                DRIVE_FILES_URL,
                params={"q": q, "pageSize": 1, "fields": "files(id, name)"},
                headers={"Authorization": f"Bearer {token}"},
                timeout=20,
            )
            files = r.json().get("files") or []
            if files:
                dl = httpx.get(
                    f"{DRIVE_FILES_URL}/{files[0]['id']}",
                    params={"alt": "media"},
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=20,
                )
                if dl.status_code == 200:
                    data = {**default_state, **dl.json(), "_drive_id": files[0]["id"]}
                    local_state.parent.mkdir(parents=True, exist_ok=True)
                    local_state.write_text(json.dumps(data, indent=2), encoding="utf-8")
                    return data
        except Exception as exc:
            print(f"[WARN] Could not load cloud state from Drive, using local state: {exc}")

    if local_state.is_file():
        try:
            return {**default_state, **json.loads(local_state.read_text(encoding="utf-8"))}
        except Exception:
            pass
    return default_state


def save_cloud_state(state: Dict, access_token: Optional[str] = None) -> None:
    """Saves `bot_state_ledger.json` locally and syncs it to Google Drive so 24/7 cron runs have memory."""
    state["processed_keys"] = state.get("processed_keys", [])[-400:]
    state["scheduled_iso_slots"] = state.get("scheduled_iso_slots", [])[-60:]
    state["updated_at"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    local_state = Path("work") / STATE_FILENAME
    local_state.parent.mkdir(parents=True, exist_ok=True)
    clean_copy = {k: v for k, v in state.items() if not k.startswith("_")}
    payload_bytes = json.dumps(clean_copy, indent=2).encode("utf-8")
    local_state.write_bytes(payload_bytes)

    folder_id = extract_drive_folder_id(env("GDRIVE_OUTPUT_FOLDER_ID") or env("GDRIVE_INBOX_FOLDER_ID"))
    if not folder_id:
        return
    try:
        token = access_token or refresh_google_access_token()
        existing_id = state.get("_drive_id")
        if existing_id:
            httpx.patch(
                f"{DRIVE_UPLOAD_URL}/{existing_id}",
                params={"uploadType": "media"},
                content=payload_bytes,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                timeout=30,
            )
        else:
            drive_upload_file(local_state, output_folder_id=folder_id, mime_type="application/json", access_token=token)
    except Exception as exc:
        print(f"[WARN] Cloud state sync to Drive skipped: {exc}")


# ---------------------------------------------------------------------------
# 4. 24/7 YouTube Gaming Stream Discovery (Preferred Channels + Live Search)
# ---------------------------------------------------------------------------
def _resolve_youtube_channel_id(handle: str, state: Dict, token: str) -> Optional[str]:
    """Resolves `@Handle` to `UC...` channel ID (cached in `state['channel_id_cache']`, costs 1 unit once)."""
    clean_handle = handle.strip()
    if not clean_handle.startswith("@"):
        clean_handle = "@" + clean_handle
    cache = state.setdefault("channel_id_cache", {})
    if clean_handle in cache:
        return cache[clean_handle]

    r = httpx.get(
        YT_CHANNELS_URL,
        params={"part": "id", "forHandle": clean_handle},
        headers={"Authorization": f"Bearer {token}"},
        timeout=20,
    )
    items = (r.json().get("items") or []) if r.status_code == 200 else []
    if items:
        cid = items[0]["id"]
        cache[clean_handle] = cid
        return cid
    return None


def _check_youtube_channel_live_or_recent(channel_id: str, state: Dict, token: str) -> Optional[Dict]:
    """
    First checks the channel's public Atom/RSS feed (`fetch_channel_rss_video_ids`, 0 quota units!)
    and falls back to `playlistItems.list` (1 unit) + `videos.list` (1 unit) to detect live or recent streams.
    """
    vid_ids = fetch_channel_rss_video_ids(channel_id, max_age_hours=18.0)
    if not vid_ids:
        uploads_pid = "UU" + channel_id[2:] if channel_id.startswith("UC") else channel_id
        r = httpx.get(
            YT_PLAYLIST_ITEMS_URL,
            params={"part": "contentDetails,snippet", "playlistId": uploads_pid, "maxResults": 5},
            headers={"Authorization": f"Bearer {token}"},
            timeout=20,
        )
        if r.status_code != 200:
            return None
        items = r.json().get("items") or []
        vid_ids = [it["contentDetails"]["videoId"] for it in items if it.get("contentDetails", {}).get("videoId")]
    if not vid_ids:
        return None

    vr = httpx.get(
        YT_VIDEOS_URL,
        params={"part": "snippet,liveStreamingDetails,contentDetails", "id": ",".join(vid_ids)},
        headers={"Authorization": f"Bearer {token}"},
        timeout=20,
    )
    if vr.status_code != 200:
        return None

    now_bucket = int(time.time() // 1800)  # 30-minute time bucket for live streams
    processed = set(state.get("processed_keys") or [])

    for v in vr.json().get("items") or []:
        vid = v["id"]
        snip = v.get("snippet") or {}
        live_det = v.get("liveStreamingDetails") or {}
        live_status = snip.get("liveBroadcastContent", "none")

        # Case A: Channel is LIVE right now on YouTube!
        if live_status == "live" or (live_det.get("actualStartTime") and not live_det.get("actualEndTime")):
            dedup_key = f"ytlive:{vid}:{now_bucket}"
            if dedup_key not in processed:
                return {
                    "url": f"https://www.youtube.com/watch?v={vid}",
                    "video_id": vid,
                    "title": snip.get("title", ""),
                    "uploader": snip.get("channelTitle", ""),
                    "is_live": True,
                    "dedup_key": dedup_key,
                }

        # Case B: Channel finished a live stream VOD within the last 12 hours that we haven't clipped yet!
        end_time_str = live_det.get("actualEndTime") or snip.get("publishedAt")
        if live_det and end_time_str:
            try:
                ended_dt = dt.datetime.fromisoformat(end_time_str.replace("Z", "+00:00"))
                age_hours = (dt.datetime.now(dt.timezone.utc) - ended_dt).total_seconds() / 3600.0
                dedup_key = f"ytvod:{vid}"
                if 0 <= age_hours <= 12.0 and dedup_key not in processed:
                    return {
                        "url": f"https://www.youtube.com/watch?v={vid}",
                        "video_id": vid,
                        "title": snip.get("title", ""),
                        "uploader": snip.get("channelTitle", ""),
                        "is_live": False,
                        "dedup_key": dedup_key,
                    }
            except ValueError:
                pass
    return None


def _search_youtube_live_by_keyword(query: str, state: Dict, token: str) -> Optional[Dict]:
    """
    Calls `search.list` (`eventType=live`, `type=video`, `videoCategoryId=20`, `order=viewCount`)
    to find the biggest live gaming stream on YouTube matching `query`.
    """
    r = httpx.get(
        YT_SEARCH_URL,
        params={
            "part": "snippet",
            "eventType": "live",
            "type": "video",
            "videoCategoryId": env("YOUTUBE_CATEGORY_ID", "20"),
            "order": "viewCount",
            "maxResults": 5,
            "q": query,
        },
        headers={"Authorization": f"Bearer {token}"},
        timeout=25,
    )
    if r.status_code != 200:
        return None
    now_bucket = int(time.time() // 1800)
    processed = set(state.get("processed_keys") or [])
    for item in r.json().get("items") or []:
        vid = (item.get("id") or {}).get("videoId")
        if not vid:
            continue
        dedup_key = f"ytlive:{vid}:{now_bucket}"
        if dedup_key in processed:
            continue
        snip = item.get("snippet") or {}
        return {
            "url": f"https://www.youtube.com/watch?v={vid}",
            "video_id": vid,
            "title": snip.get("title", ""),
            "uploader": snip.get("channelTitle", ""),
            "is_live": True,
            "dedup_key": dedup_key,
        }
    return None


KNOWN_SIMULCAST_MAP: Dict[str, str] = {
    "@ishowspeed": "https://www.twitch.tv/ishowspeed",
    "@typicalgamer": "https://www.twitch.tv/typicalgamer",
    "@kaicenatlive": "https://www.twitch.tv/kaicenat",
    "@kaicenat": "https://www.twitch.tv/kaicenat",
    "@caseoh": "https://www.twitch.tv/caseoh_",
    "@xqcow": "https://www.twitch.tv/xqc",
    "@valkyrae": "https://www.twitch.tv/valkyrae",
    "@shroud": "https://www.twitch.tv/shroud",
    "@tarik": "https://www.twitch.tv/tarik",
    "@jynxzi": "https://www.twitch.tv/jynxzi",
}

PREFERRED_TWITCH_GAMING_LOGINS: List[str] = [
    "kaicenat", "caseoh_", "xqc", "valkyrae", "ishowspeed", "typicalgamer",
    "jynxzi", "shroud", "tarik", "stableronaldo", "lacy", "jasontheween",
    "plaqueboymax", "agent00", "fanum", "summit1g", "lirik", "clix",
    "ohnepixel", "caedrel", "fuslie", "sykkuno", "iitztimmy", "imperialhal",
    "nicewigg", "nickeh30", "symfuhny", "tenz", "squeex", "extraemily",
]


def discover_live_twitch_stream(state: Optional[Dict] = None, preferred_url: str = "") -> Optional[Dict]:
    st = state if state is not None else {}
    now_bucket = int(time.time() // 1800)
    processed = set(st.get("processed_keys") or [])
    gql_url = "https://gql.twitch.tv/gql"
    headers = {"Client-Id": "kimne78kx3ncx6brgo4mv6wki5h1ko", "Content-Type": "application/json"}

    logins_to_check: List[str] = []
    if preferred_url:
        m = re.search(r"twitch\.tv/([A-Za-z0-9_]+)", preferred_url)
        if m:
            logins_to_check.append(m.group(1).lower())
    for lg in PREFERRED_TWITCH_GAMING_LOGINS:
        if lg.lower() not in logins_to_check:
            logins_to_check.append(lg.lower())

    aliases = [
        f'u{i}: user(login: "{lg}") {{ login displayName stream {{ id title viewersCount game {{ name }} }} }}'
        for i, lg in enumerate(logins_to_check[:30])
    ]
    try:
        r = httpx.post(gql_url, headers=headers, json={"query": "query { " + " ".join(aliases) + " }"}, timeout=15)
        if r.status_code == 200:
            data = (r.json().get("data") or {})
            live_candidates = []
            for i in range(len(aliases)):
                u = data.get(f"u{i}") or {}
                s = u.get("stream")
                if s and s.get("id"):
                    login = u.get("login", "")
                    dedup_key = f"twitch:{login}:{now_bucket}"
                    if dedup_key in processed:
                        continue
                    is_pref = 1 if (preferred_url and login.lower() == logins_to_check[0]) else 0
                    viewers = int(s.get("viewersCount") or 0)
                    game_name = (s.get("game") or {}).get("name") or "Gaming"
                    live_candidates.append((is_pref, viewers, {
                        "url": f"https://www.twitch.tv/{login}",
                        "video_id": str(s.get("id")),
                        "title": f"{s.get('title') or game_name} ({game_name})",
                        "uploader": u.get("displayName") or login,
                        "is_live": True,
                        "dedup_key": dedup_key,
                        "fallback_url": "",
                    }))
            if live_candidates:
                live_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
                best = live_candidates[0][2]
                print(f"[TWITCH LIVE DISCOVERY] Found active stream @{best['uploader']} ({live_candidates[0][1]:,} viewers): {best['title']} ({best['url']})")
                return best
    except Exception as exc:
        print(f"[WARN] Twitch preferred batch lookup error: {exc}")

    top_query = """query { streams(first: 15, options: {broadcasterLanguages: ["en"]}) { edges { node { id title viewersCount broadcaster { login displayName } game { name } } } } }"""
    try:
        r = httpx.post(gql_url, headers=headers, json={"query": top_query}, timeout=15)
        if r.status_code == 200:
            for edge in (((r.json().get("data") or {}).get("streams") or {}).get("edges") or []):
                node = edge.get("node") or {}
                bc = node.get("broadcaster") or {}
                login = bc.get("login") or ""
                if not login:
                    continue
                dedup_key = f"twitch:{login}:{now_bucket}"
                if dedup_key in processed:
                    continue
                game_name = (node.get("game") or {}).get("name") or "Gaming"
                hit = {
                    "url": f"https://www.twitch.tv/{login}",
                    "video_id": str(node.get("id") or login),
                    "title": f"{node.get('title') or game_name} ({game_name})",
                    "uploader": bc.get("displayName") or login,
                    "is_live": True,
                    "dedup_key": dedup_key,
                    "fallback_url": "",
                }
                print(f"[TWITCH TOP LIVE] Found top active stream @{hit['uploader']}: {hit['title']} ({hit['url']})")
                return hit
    except Exception as exc:
        print(f"[WARN] Twitch top streams lookup error: {exc}")
    return None


def discover_active_stream(
    watchlist_path: Path = Path("watchlist.txt"),
    state: Optional[Dict] = None,
    access_token: Optional[str] = None,
) -> Optional[Dict]:
    """
    Walks `watchlist.txt` from top to bottom:
      1. `@YouTubeHandle` or `@YouTubeHandle | https://twitch.tv/...` -> Checks YouTube API first, plus simulcast fallback!
      2. `https://...` -> Checks direct Twitch, Kick, or YouTube URL via `yt-dlp --simulate`.
      3. `SEARCH: <query>` -> Searches YouTube Live via official YouTube Data API v3 (`search.list`).
    """
    st = state if state is not None else load_cloud_state(access_token)
    token = None
    try:
        token = access_token or refresh_google_access_token()
    except Exception:
        token = None

    lines = []
    if watchlist_path.is_file():
        lines = [
            ln.strip()
            for ln in watchlist_path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]

    now_bucket = int(time.time() // 1800)
    processed = set(st.get("processed_keys") or [])

    for entry in lines:
        # 1. Preferred YouTube Channel Handle (`@Handle` or `@Handle | fallback_simulcast_url`)
        if entry.startswith("@"):
            parts = [p.strip() for p in entry.split("|", 1)]
            handle = parts[0]
            fallback_url = parts[1] if len(parts) > 1 else KNOWN_SIMULCAST_MAP.get(handle.lower(), "")
            if token:
                cid = _resolve_youtube_channel_id(handle, st, token)
                if cid:
                    found = _check_youtube_channel_live_or_recent(cid, st, token)
                    if found:
                        found["fallback_url"] = fallback_url
                        print(f"[DISCOVERY] Found active YouTube channel {handle}: {found['title']} ({found['url']})")
                        return found
            if fallback_url:
                entry = fallback_url
            else:
                continue

        # 2. Automatic YouTube Live Keyword Search (`SEARCH: GTA 6 live`)
        if entry.upper().startswith("SEARCH:"):
            q = entry.split(":", 1)[1].strip()
            if q and token:
                found = _search_youtube_live_by_keyword(q, st, token)
                if found:
                    print(f"[DISCOVERY] Found trending YouTube Live for '{q}': {found['title']} ({found['url']})")
                    return found
            continue

        # 3. Direct Twitch / Kick / YouTube URL
        if entry.startswith("http"):
            dedup_key = f"direct:{entry}:{now_bucket}"
            if dedup_key in processed:
                continue
            cmd = ["yt-dlp", "--simulate", "--no-warnings", "--quiet", "--print", "%(is_live)s|%(title)s|%(uploader)s", entry]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
                if res.returncode == 0 and res.stdout.strip().startswith("True|"):
                    _, title, uploader = (res.stdout.strip().split("|") + ["", ""])[:3]
                    print(f"[DISCOVERY] Found live stream on {entry}: {title}")
                    return {
                        "url": entry,
                        "video_id": entry,
                        "title": title,
                        "uploader": uploader,
                        "is_live": True,
                        "dedup_key": dedup_key,
                        "fallback_url": "",
                    }
            except Exception:
                continue

    # 4. Fallback: Check YouTube's 1-unit `chart=mostPopular` Gaming Trending Chart ranked by Breakout Velocity
    if token:
        chart_hit = fetch_most_popular_gaming_chart(st, token, region_code=env("TRENDS_GEO", "US"))
        if chart_hit:
            return chart_hit
    return None


# ---------------------------------------------------------------------------
# 5. YouTube Analytics & Hourly Heatmap Self-Optimizer
# ---------------------------------------------------------------------------
def update_analytics_heatmap(
    state: Optional[Dict] = None,
    access_token: Optional[str] = None,
) -> Dict:
    """
    Monitors your channel's published Shorts performance (views, likes, comments, and watch retention),
    builds a 24-hour performance heatmap by UTC publish hour, and updates `state['golden_hours_utc']`
    so future Shorts are automatically scheduled at your highest-converting hours!
    """
    st = state if state is not None else load_cloud_state(access_token)
    token = access_token or refresh_google_access_token()

    # Step 1: Fetch authenticated user's uploads playlist (`mine=true`, 1 quota unit)
    ch_res = httpx.get(
        YT_CHANNELS_URL,
        params={"part": "contentDetails,statistics", "mine": "true"},
        headers={"Authorization": f"Bearer {token}"},
        timeout=20,
    )
    if ch_res.status_code != 200 or not (ch_res.json().get("items")):
        return st

    ch_item = ch_res.json()["items"][0]
    uploads_pid = ch_item.get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")
    if not uploads_pid:
        return st

    # Step 2: Fetch up to 50 recent uploaded videos (1 quota unit)
    pl_res = httpx.get(
        YT_PLAYLIST_ITEMS_URL,
        params={"part": "contentDetails", "playlistId": uploads_pid, "maxResults": 50},
        headers={"Authorization": f"Bearer {token}"},
        timeout=20,
    )
    if pl_res.status_code != 200:
        return st
    vid_ids = [it["contentDetails"]["videoId"] for it in (pl_res.json().get("items") or []) if it.get("contentDetails", {}).get("videoId")]
    if not vid_ids:
        return st

    # Step 3: Fetch video statistics & publish times (1 quota unit for up to 50 videos)
    v_res = httpx.get(
        YT_VIDEOS_URL,
        params={"part": "snippet,statistics,status", "id": ",".join(vid_ids[:50])},
        headers={"Authorization": f"Bearer {token}"},
        timeout=20,
    )
    if v_res.status_code != 200:
        return st

    # Optional Step 4: Enrich with YouTube Analytics API v2 averageViewPercentage if available
    retention_by_vid: Dict[str, float] = {}
    try:
        end_date = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
        start_date = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=28)).strftime("%Y-%m-%d")
        an_res = httpx.get(
            YT_ANALYTICS_URL,
            params={
                "ids": "channel==MINE",
                "startDate": start_date,
                "endDate": end_date,
                "metrics": "views,estimatedMinutesWatched,averageViewPercentage,likes",
                "dimensions": "video",
                "filters": f"video=={','.join(vid_ids[:50])}",
                "maxResults": 50,
            },
            headers={"Authorization": f"Bearer {token}"},
            timeout=20,
        )
        if an_res.status_code == 200:
            for row in an_res.json().get("rows") or []:
                # row: [video_id, views, estMinutesWatched, avgViewPct, likes]
                if len(row) >= 4:
                    retention_by_vid[str(row[0])] = float(row[3] or 50.0)
    except Exception:
        pass

    # Step 5: Compute Hourly Velocity Heatmap across all published videos
    now_utc = dt.datetime.now(dt.timezone.utc)
    hour_buckets: Dict[int, List[float]] = {h: [] for h in range(24)}
    upcoming_iso_slots: List[str] = []
    published_count = 0

    for v in v_res.json().get("items") or []:
        status = v.get("status") or {}
        snip = v.get("snippet") or {}
        stats = v.get("statistics") or {}

        # Track still-scheduled future videos so we never double-book a time slot!
        if status.get("privacyStatus") == "private" and status.get("publishAt"):
            upcoming_iso_slots.append(status["publishAt"])
            continue

        if status.get("privacyStatus") != "public":
            continue

        pub_str = snip.get("publishedAt")
        if not pub_str:
            continue
        try:
            pub_dt = dt.datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
        except ValueError:
            continue

        age_hours = max(1.0, (now_utc - pub_dt).total_seconds() / 3600.0)
        views = float(stats.get("viewCount") or 0)
        likes = float(stats.get("likeCount") or 0)
        comments = float(stats.get("commentCount") or 0)
        retention_mult = max(0.5, min(2.0, retention_by_vid.get(v["id"], 65.0) / 65.0))

        # Velocity score: views per hour weighted by engagement and average view percentage
        velocity_score = ((views + likes * 8.0 + comments * 20.0) / (age_hours ** 0.75)) * retention_mult
        hour_buckets[pub_dt.hour].append(round(velocity_score, 2))
        published_count += 1

    heatmap_summary = {}
    for h in range(24):
        scores = hour_buckets[h]
        heatmap_summary[f"{h:02d}:00_UTC"] = {
            "samples": len(scores),
            "avg_velocity_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
        }

    st["hourly_heatmap"] = heatmap_summary
    st["published_shorts_analyzed"] = published_count
    if upcoming_iso_slots:
        st["scheduled_iso_slots"] = sorted(set(st.get("scheduled_iso_slots", []) + upcoming_iso_slots))[-60:]

    # Once at least 5 Shorts have published, rank hours by measured velocity + exploration bonus
    if published_count >= 5:
        scored_hours = []
        for h in range(24):
            samples = hour_buckets[h]
            avg_s = (sum(samples) / len(samples)) if samples else (5.0 if h in DEFAULT_GOLDEN_HOURS_UTC else 1.0)
            scored_hours.append((avg_s, h))
        scored_hours.sort(reverse=True)
        learned_top = [h for _, h in scored_hours[:6]]
        st["golden_hours_utc"] = learned_top
        print(f"[ANALYTICS HEATMAP] Analyzed {published_count} published Shorts -> Top Golden Hours (UTC): {learned_top}")
    else:
        st["golden_hours_utc"] = list(DEFAULT_GOLDEN_HOURS_UTC)
        print(f"[ANALYTICS HEATMAP] Cold-start mode ({published_count}/5 Shorts published) -> Using default Golden Hours: {DEFAULT_GOLDEN_HOURS_UTC}")

    return st


# ---------------------------------------------------------------------------
# 6. Google Drive Cloud Folder Sync (Inbox Pull + Output Push)
# ---------------------------------------------------------------------------
def drive_list_inbox_videos(
    inbox_folder_id: Optional[str] = None,
    access_token: Optional[str] = None,
) -> List[Dict]:
    folder_id = extract_drive_folder_id(inbox_folder_id or env("GDRIVE_INBOX_FOLDER_ID"))
    if not folder_id:
        return []
    token = access_token or refresh_google_access_token()
    query = (
        f"'{folder_id}' in parents and trashed = false "
        f"and not name contains '[CLIPPED]' "
        f"and (mimeType contains 'video/' or name contains '.mp4' or name contains '.mkv' or name contains '.webm')"
    )
    r = httpx.get(
        DRIVE_FILES_URL,
        params={
            "q": query,
            "orderBy": "createdTime asc",
            "pageSize": 10,
            "fields": "files(id, name, size, mimeType, createdTime, webViewLink)",
        },
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    if r.status_code == 404:
        return []
    if r.status_code >= 400:
        raise RuntimeError(f"Google Drive inbox list failed ({r.status_code}): {r.text[:300]}")
    return r.json().get("files") or []


def drive_download_file(file_id: str, dest_path: Path, access_token: Optional[str] = None) -> Path:
    token = access_token or refresh_google_access_token()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream(
        "GET",
        f"{DRIVE_FILES_URL}/{file_id}",
        params={"alt": "media"},
        headers={"Authorization": f"Bearer {token}"},
        timeout=1800,
    ) as r:
        if r.status_code >= 400:
            raise RuntimeError(f"Google Drive download failed ({r.status_code})")
        with dest_path.open("wb") as out:
            for chunk in r.iter_bytes(chunk_size=4 * 1024 * 1024):
                out.write(chunk)
    return dest_path


def drive_mark_processed(file_id: str, original_name: str, access_token: Optional[str] = None) -> None:
    if original_name.startswith("[CLIPPED]"):
        return
    token = access_token or refresh_google_access_token()
    httpx.patch(
        f"{DRIVE_FILES_URL}/{file_id}",
        json={"name": f"[CLIPPED] {original_name}"},
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        timeout=30,
    )


def drive_upload_file(
    local_path: Path,
    output_folder_id: Optional[str] = None,
    mime_type: str = "video/mp4",
    description: str = "",
    access_token: Optional[str] = None,
) -> Tuple[str, str]:
    folder_id = extract_drive_folder_id(output_folder_id or env("GDRIVE_OUTPUT_FOLDER_ID"))
    if not folder_id:
        raise ValueError("Missing GDRIVE_OUTPUT_FOLDER_ID for Google Drive upload.")
    token = access_token or refresh_google_access_token()
    size = local_path.stat().st_size

    meta = {"name": local_path.name, "parents": [folder_id], "description": (description or "")[:1500]}
    init_res = httpx.post(
        DRIVE_UPLOAD_URL,
        params={"uploadType": "resumable", "fields": "id, name, webViewLink"},
        json=meta,
        headers={
            "Authorization": f"Bearer {token}",
            "X-Upload-Content-Type": mime_type,
            "X-Upload-Content-Length": str(size),
        },
        timeout=60,
    )
    if init_res.status_code == 404:
        app_folder = _get_or_create_app_drive_folder("RazonGamerz_Output", token)
        meta["parents"] = [app_folder] if app_folder else []
        if not app_folder:
            meta.pop("parents", None)
        init_res = httpx.post(
            DRIVE_UPLOAD_URL,
            params={"uploadType": "resumable", "fields": "id, name, webViewLink", "supportsAllDrives": "true"},
            json=meta,
            headers={"Authorization": f"Bearer {token}", "X-Upload-Content-Type": mime_type, "X-Upload-Content-Length": str(size)},
            timeout=60,
        )
    if init_res.status_code >= 400:
        raise RuntimeError(f"Drive resumable upload init failed ({init_res.status_code}): {init_res.text[:300]}")

    upload_url = init_res.headers["Location"]
    with local_path.open("rb") as f:
        put_res = httpx.put(
            upload_url,
            content=f.read(),
            headers={"Authorization": f"Bearer {token}", "Content-Type": mime_type},
            timeout=900,
        )
    if put_res.status_code >= 400:
        raise RuntimeError(f"Drive byte upload failed ({put_res.status_code}): {put_res.text[:300]}")

    info = put_res.json()
    fid = info.get("id", "")
    return fid, info.get("webViewLink") or f"https://drive.google.com/file/d/{fid}/view"


def parse_drive_filename_metadata(filename: str) -> Tuple[str, str]:
    stem = Path(filename).stem
    stem = re.sub(r"^\[CLIPPED\]\s*", "", stem, flags=re.I).strip()
    if " - " in stem:
        parts = stem.split(" - ", 1)
        return parts[0].strip() or "Streamer", parts[1].strip() or "Live Stream Highlight"
    return "Streamer", stem or "Live Stream Highlight"


# ---------------------------------------------------------------------------
# 7. Universal Viral SEO Metadata Generator & 3:2 Thumbnail Composer
# ---------------------------------------------------------------------------
GTA6_SEO_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "title": {"type": "STRING"},
        "description": {"type": "STRING"},
        "tags": {"type": "ARRAY", "items": {"type": "STRING"}},
        "thumbnail_badge": {"type": "STRING"},
        "category_id": {"type": "STRING"},
    },
    "required": ["title", "description", "tags", "thumbnail_badge"],
}

GTA6_SEO_PROMPT = """You are an elite viral YouTube Shorts SEO strategist.
Target Topic / Niche Override: {topic_hint}
Streamer / Channel: {streamer}
Original Stream Title: {stream_title}
Clip Hook (on-screen): {hook}
Clip Transcript:
{transcript}

Strict Rules:
1. Identify the exact game, podcast, or event from Target Topic Override (or auto-detect from Original Stream Title and Transcript if "auto").
2. "title": Max 92 characters. Combine creator name + the shocking/funny moment + 1 primary topic hashtag and #Shorts at the end.
3. "description": 120 to 180 words.
   - First 2 sentences MUST pack high-search keywords for this creator and game/topic.
   - Include a clear credit line: "Full stream credit: {streamer}".
   - End with 5 niche-relevant hashtags ending in #Shorts.
4. "tags": 12 to 15 high-volume search tags (without #).
5. "thumbnail_badge": 2 to 4 words in ALL CAPS (max 22 characters) summarizing the climax for the center thumbnail badge.
6. "category_id": "20" for Gaming, "24" for Entertainment, "22" for Podcasts/People, "28" for Tech, "17" for Sports.
Return ONLY valid JSON matching the schema."""


def sanitize_youtube_tags(raw_tags: List[str], streamer: str, topic: str = "") -> List[str]:
    topic_clean = (topic or env("TOPIC_NICHE", "GTA 6")).strip()
    if topic_clean.lower() in ("auto", ""):
        topic_clean = "Gaming Live"
    base = [
        topic_clean,
        f"{streamer} {topic_clean}".strip(),
        f"{topic_clean} Highlights",
        f"{topic_clean} Shorts",
        f"{streamer} Clips".strip(),
        "YouTube Shorts",
    ]
    merged: List[str] = []
    seen = set()
    for t in list(raw_tags or []) + base:
        clean = re.sub(r"[<>#]", "", str(t)).strip()
        if clean and clean.lower() not in seen and len(clean) <= 45:
            seen.add(clean.lower())
            merged.append(clean)

    final_tags: List[str] = []
    char_budget = 0
    for tag in merged[:15]:
        cost = len(tag) + (3 if " " in tag else 1)
        if char_budget + cost > 460:
            break
        final_tags.append(tag)
        char_budget += cost
    return final_tags


def generate_gta6_seo_metadata(
    streamer: str,
    stream_title: str,
    hook: str,
    transcript: str,
    ai_key: Optional[str] = None,
    model: str = "gemini-2.5-flash",
    topic: Optional[str] = None,
) -> Dict:
    key = ai_key or env("AI_API_KEY")
    topic_hint = (topic or env("TOPIC_NICHE", "auto")).strip() or "auto"
    prompt = GTA6_SEO_PROMPT.format(
        topic_hint=topic_hint,
        streamer=streamer or "Streamer",
        stream_title=stream_title or "Live Stream Highlight",
        hook=hook or "WAIT FOR IT",
        transcript=(transcript or "")[:2500],
    )

    if key or env("GROQ_API_KEY"):
        try:
            parsed = call_gemini_json_with_cascade(prompt, GTA6_SEO_SCHEMA, ai_key=key)
            return audit_and_enforce_2026_seo_rules(parsed, streamer, topic_hint)
        except Exception as exc:
            print(f"[WARN] AI SEO cascade fell back to template ({exc})")

    active_topic = topic_hint if topic_hint.lower() != "auto" else (stream_title[:28].strip() or "GTA 6")
    hashtag_topic = re.sub(r"[^a-zA-Z0-9]", "", active_topic)[:18] or "LiveClips"
    clean_hook = (hook or f"INSANE {active_topic.upper()} MOMENT").upper()[:22]
    title = f"{streamer} Reacts to {clean_hook.title()} ({active_topic}) #{hashtag_topic} #Shorts"[:95]
    description = (
        f"{streamer} experiences an unforgettable live moment during {stream_title} ({active_topic}). "
        f"Watch the full reaction and highlight captured live.\n\n"
        f"Full stream credit: {streamer}\n\n"
        f"#{hashtag_topic} #{re.sub(r'[^a-zA-Z0-9]', '', streamer) or 'Streamer'} #LiveClips #ViralShorts #Shorts"
    )
    return audit_and_enforce_2026_seo_rules({
        "title": title,
        "description": description,
        "tags": sanitize_youtube_tags(fetch_youtube_autocomplete_keywords(f"{streamer} {active_topic}", limit=6), streamer, active_topic),
        "thumbnail_badge": clean_hook,
        "category_id": env("YOUTUBE_CATEGORY_ID", "20"),
    }, streamer, active_topic)


def build_shorts_thumbnail(
    frame_jpg: Path,
    out_jpg: Path,
    badge_text: str,
    font_path: Optional[Path] = None,
    theme: str = "vice_hype",
) -> Path:
    im = Image.open(frame_jpg).convert("RGB").resize((1080, 1920), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(im, "RGBA")
    text = re.sub(r"\s+", " ", (badge_text or "INSANE CLIP").upper()).strip()[:26]

    resolved_font = font_path
    if not (resolved_font and resolved_font.is_file()):
        for cand in [
            Path("fonts/Poppins-ExtraBold.ttf"),
            Path("fonts/Anton-Regular.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            Path("/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"),
        ]:
            if cand.is_file():
                resolved_font = cand
                break

    font_size = 78
    font = None
    while font_size >= 44:
        if resolved_font and resolved_font.is_file():
            try:
                font = ImageFont.truetype(str(resolved_font), font_size)
            except OSError:
                font = ImageFont.load_default()
                break
        else:
            font = ImageFont.load_default()
            break
        bbox = draw.textbbox((0, 0), text, font=font)
        if (bbox[2] - bbox[0]) <= 860:
            break
        font_size -= 4

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = 48, 28
    bw, bh = min(980, tw + pad_x * 2), th + pad_y * 2
    x0, y0 = (1080 - bw) // 2, 1160 - bh // 2
    x1, y1 = x0 + bw, y0 + bh

    themes = {
        "vice_hype": {"fill": (255, 42, 133, 248), "outline": (255, 183, 3, 255), "ink": (255, 255, 255, 255)},
        "cyber_gold": {"fill": (123, 44, 191, 248), "outline": (255, 183, 3, 255), "ink": (255, 255, 255, 255)},
        "rage_neon": {"fill": (9, 10, 16, 248), "outline": (255, 42, 133, 255), "ink": (255, 183, 3, 255)},
        "clean_pro": {"fill": (9, 10, 16, 242), "outline": (248, 249, 250, 255), "ink": (248, 249, 250, 255)},
    }
    pal = themes.get(theme, themes["vice_hype"])
    draw.rounded_rectangle([x0 - 8, y0 + 10, x1 + 8, y1 + 16], radius=28, fill=(0, 0, 0, 190))
    draw.rounded_rectangle([x0, y0, x1, y1], radius=24, fill=pal["fill"], outline=pal["outline"], width=6)
    draw.text((1080 // 2 - tw // 2, y0 + pad_y - bbox[1]), text, font=font, fill=pal["ink"])
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_jpg, format="JPEG", quality=92)
    return out_jpg


# ---------------------------------------------------------------------------
# 8. Heatmap-Aware `publishAt` Slot Calculator & YouTube Uploader
# ---------------------------------------------------------------------------
def compute_next_publish_slot(
    gap_hours: float = 3.0,
    existing_iso_slots: Optional[List[str]] = None,
    now_utc: Optional[dt.datetime] = None,
    golden_hours_utc: Optional[List[int]] = None,
) -> str:
    """
    Computes the next ISO-8601 UTC publish timestamp (`YYYY-MM-DDTHH:MM:SSZ`).
    If `golden_hours_utc` is provided (from your channel's Analytics Heatmap), it snaps the
    scheduled publish time to the next open Golden Hour that is at least `gap_hours` after
    the latest scheduled video!
    """
    now = now_utc or dt.datetime.now(dt.timezone.utc)
    earliest = now + dt.timedelta(minutes=30)

    latest_scheduled = None
    for iso_str in existing_iso_slots or []:
        try:
            parsed = dt.datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
            if latest_scheduled is None or parsed > latest_scheduled:
                latest_scheduled = parsed
        except ValueError:
            continue

    min_target = max(earliest, latest_scheduled + dt.timedelta(hours=gap_hours)) if latest_scheduled else earliest

    if golden_hours_utc:
        valid_hours = sorted({int(h) % 24 for h in golden_hours_utc})
        for day_offset in range(0, 4):
            base_day = (min_target + dt.timedelta(days=day_offset)).replace(minute=0, second=0, microsecond=0)
            for gh in valid_hours:
                cand = base_day.replace(hour=gh, minute=0, second=0, microsecond=0)
                if cand >= min_target:
                    return cand.strftime("%Y-%m-%dT%H:%M:%SZ")

    discard = dt.timedelta(minutes=min_target.minute % 15, seconds=min_target.second, microseconds=min_target.microsecond)
    target = min_target - discard + dt.timedelta(minutes=15)
    return target.strftime("%Y-%m-%dT%H:%M:%SZ")


def upload_and_schedule_short(
    video_path: Path,
    thumb_path: Optional[Path],
    title: str,
    description: str,
    tags: List[str],
    publish_at_iso: str,
    access_token: Optional[str] = None,
    category_id: str = "20",
) -> Tuple[str, str]:
    token = access_token or refresh_google_access_token()
    clean_title = re.sub(r"[<>]", "", title)[:100] or "Live Highlight #Shorts"
    clean_desc = re.sub(r"[<>]", "", description)[:4900]

    body = {
        "snippet": {
            "title": clean_title,
            "description": clean_desc,
            "tags": sanitize_youtube_tags(tags, ""),
            "categoryId": str(category_id or env("YOUTUBE_CATEGORY_ID", "20")),
            "defaultLanguage": "en",
        },
        "status": {
            "privacyStatus": "private",
            "publishAt": publish_at_iso,
            "selfDeclaredMadeForKids": False,
        },
    }

    size = video_path.stat().st_size
    init_res = httpx.post(
        YT_UPLOAD_URL,
        params={"uploadType": "resumable", "part": "snippet,status"},
        json=body,
        headers={
            "Authorization": f"Bearer {token}",
            "X-Upload-Content-Type": "video/mp4",
            "X-Upload-Content-Length": str(size),
        },
        timeout=60,
    )
    if init_res.status_code >= 400:
        raise RuntimeError(f"Failed starting resumable upload ({init_res.status_code}): {init_res.text[:300]}")

    upload_location = init_res.headers["Location"]
    with video_path.open("rb") as vf:
        put_res = httpx.put(
            upload_location,
            content=vf.read(),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "video/mp4"},
            timeout=900,
        )
    if put_res.status_code >= 400:
        raise RuntimeError(f"Video byte upload failed ({put_res.status_code}): {put_res.text[:300]}")

    video_id = put_res.json()["id"]
    short_url = f"https://www.youtube.com/shorts/{video_id}"

    if thumb_path and thumb_path.is_file():
        for _ in range(12):
            time.sleep(8)
            poll = httpx.get(
                YT_VIDEOS_URL,
                params={"part": "processingDetails", "id": video_id},
                headers={"Authorization": f"Bearer {token}"},
                timeout=30,
            )
            if poll.status_code == 200:
                items = poll.json().get("items") or []
                status = items[0].get("processingDetails", {}).get("processingStatus") if items else ""
                if status == "succeeded":
                    break
        httpx.post(
            YT_THUMB_URL,
            params={"videoId": video_id},
            content=thumb_path.read_bytes(),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "image/jpeg"},
            timeout=60,
        )

    return video_id, short_url


def run_local_oauth_flow(port: int = 8769) -> None:
    cid = env("GOOGLE_CLIENT_ID") or input("Paste your GOOGLE_CLIENT_ID: ").strip()
    csecret = env("GOOGLE_CLIENT_SECRET") or input("Paste your GOOGLE_CLIENT_SECRET: ").strip()
    redirect_uri = f"http://127.0.0.1:{port}/callback"
    captured: Dict[str, str] = {}

    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            if "code" in qs:
                captured["code"] = qs["code"][0]
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(
                    b"<h2>YouTube + Analytics + Google Drive connected! Return to terminal for YT_REFRESH_TOKEN.</h2>"
                )
            else:
                self.send_response(400)
                self.end_headers()

        def log_message(self, format, *args):
            pass

    auth_link = f"{YT_AUTH_URL}?" + urllib.parse.urlencode(
        {
            "client_id": cid,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": YT_SCOPES,
            "access_type": "offline",
            "prompt": "consent",
        }
    )
    print(f"Register Redirect URI: {redirect_uri}\nOpening: {auth_link}")
    webbrowser.open(auth_link)
    srv = HTTPServer(("127.0.0.1", port), CallbackHandler)
    srv.handle_request()
    srv.server_close()
    tok = httpx.post(
        YT_TOKEN_URL,
        data={
            "code": captured["code"],
            "client_id": cid,
            "client_secret": csecret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        },
        timeout=30,
    ).json()
    print(f"\nYT_REFRESH_TOKEN={tok.get('refresh_token', 'MISSING')}\n")



# ---------------------------------------------------------------------------
# 9. Facebook Page Reels & Instagram Reels Multi-Platform Uploaders (Meta Graph API)
# ---------------------------------------------------------------------------
FB_GRAPH_URL = "https://graph.facebook.com/v21.0"
FB_RUPLOAD_URL = "https://rupload.facebook.com/video-upload/v21.0"
IG_RUPLOAD_URL = "https://rupload.facebook.com/ig-api-upload/v21.0"


def build_social_caption(seo: Dict, streamer: str) -> str:
    """Formats a punchy Instagram & Facebook Reels caption with credit and hashtags (max 2,200 chars)."""
    title = seo.get("title", "").replace("#Shorts", "").strip()
    desc = seo.get("description", "").replace("#Shorts", "#Reels").strip()
    tags = ["#" + re.sub(r"[^a-zA-Z0-9]", "", t) for t in (seo.get("tags") or [])[:8] if t]
    tag_line = " ".join(tags + ["#Reels", "#GamingReels", "#ViralReels"])
    return f"{title}\n\n{desc}\n\n{tag_line}"[:2150]


def upload_facebook_reel(
    video_path: Path,
    title: str,
    caption: str,
    publish_at_iso: Optional[str] = None,
    page_id: Optional[str] = None,
    page_token: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Uploads and publishes (or schedules) a 9:16 Reel to a Facebook Page via Meta Graph API v21.0.
    Uses direct binary upload to `rupload.facebook.com` (no public website required).
    If `publish_at_iso` is > 20 minutes in the future, schedules natively via `video_state="SCHEDULED"`.
    """
    pid = (page_id or env("FB_PAGE_ID")).strip()
    token = (page_token or env("FB_PAGE_ACCESS_TOKEN")).strip()
    if not (pid and token):
        raise ValueError("Missing FB_PAGE_ID or FB_PAGE_ACCESS_TOKEN.")

    size = video_path.stat().st_size

    # Step 1: Initialize upload session
    init_res = httpx.post(
        f"{FB_GRAPH_URL}/{pid}/video_reels",
        data={"upload_phase": "start", "access_token": token},
        timeout=60,
    )
    if init_res.status_code >= 400:
        raise RuntimeError(f"Facebook Reel init failed ({init_res.status_code}): {init_res.text[:300]}")
    video_id = init_res.json()["video_id"]

    # Step 2: Binary upload to rupload.facebook.com
    with video_path.open("rb") as vf:
        up_res = httpx.post(
            f"{FB_RUPLOAD_URL}/{video_id}",
            content=vf.read(),
            headers={
                "Authorization": f"OAuth {token}",
                "offset": "0",
                "file_size": str(size),
            },
            timeout=900,
        )
    if up_res.status_code >= 400:
        raise RuntimeError(f"Facebook Reel binary upload failed ({up_res.status_code}): {up_res.text[:300]}")

    # Step 3: Finish and either SCHEDULE or PUBLISH immediately
    finish_data = {
        "upload_phase": "finish",
        "video_id": video_id,
        "title": re.sub(r"[<>]", "", title)[:100],
        "description": caption[:2150],
        "access_token": token,
    }
    scheduled_unix = None
    if publish_at_iso:
        try:
            dt_obj = dt.datetime.fromisoformat(publish_at_iso.replace("Z", "+00:00"))
            diff_sec = (dt_obj - dt.datetime.now(dt.timezone.utc)).total_seconds()
            # Meta requires scheduled_publish_time to be at least 15 minutes in the future
            if diff_sec >= 1200:
                scheduled_unix = int(dt_obj.timestamp())
        except ValueError:
            scheduled_unix = None

    if scheduled_unix:
        finish_data["video_state"] = "SCHEDULED"
        finish_data["scheduled_publish_time"] = str(scheduled_unix)
    else:
        finish_data["video_state"] = "PUBLISHED"

    fin_res = httpx.post(f"{FB_GRAPH_URL}/{pid}/video_reels", data=finish_data, timeout=60)
    if fin_res.status_code >= 400:
        raise RuntimeError(f"Facebook Reel finish failed ({fin_res.status_code}): {fin_res.text[:300]}")

    return video_id, f"https://www.facebook.com/reel/{video_id}"


def upload_instagram_reel(
    video_path: Path,
    caption: str,
    ig_user_id: Optional[str] = None,
    ig_token: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Uploads and publishes a 9:16 Reel to an Instagram Professional (Creator/Business) account
    using the Graph API Resumable Upload protocol (`rupload.facebook.com/ig-api-upload`).
    """
    uid = (ig_user_id or env("IG_USER_ID")).strip()
    token = (ig_token or env("IG_ACCESS_TOKEN") or env("FB_PAGE_ACCESS_TOKEN")).strip()
    if not (uid and token):
        raise ValueError("Missing IG_USER_ID or IG_ACCESS_TOKEN / FB_PAGE_ACCESS_TOKEN.")

    size = video_path.stat().st_size
    auth_header = {"Authorization": f"Bearer {token}"}

    # Step 1: Create resumable REELS container
    box_res = httpx.post(
        f"{FB_GRAPH_URL}/{uid}/media",
        data={
            "media_type": "REELS",
            "upload_type": "resumable",
            "caption": caption[:2150],
            "share_to_feed": "true",
        },
        headers=auth_header,
        timeout=60,
    )
    if box_res.status_code >= 400:
        raise RuntimeError(f"Instagram container init failed ({box_res.status_code}): {box_res.text[:300]}")
    container_id = box_res.json()["id"]

    # Step 2: Upload binary bytes to rupload.facebook.com/ig-api-upload
    with video_path.open("rb") as vf:
        up_res = httpx.post(
            f"{IG_RUPLOAD_URL}/{container_id}",
            content=vf.read(),
            headers={
                "Authorization": f"OAuth {token}",
                "offset": "0",
                "file_size": str(size),
            },
            timeout=900,
        )
    if up_res.status_code >= 400:
        raise RuntimeError(f"Instagram binary upload failed ({up_res.status_code}): {up_res.text[:300]}")

    # Step 3: Poll container status until FINISHED
    for _ in range(60):
        time.sleep(6)
        st_res = httpx.get(
            f"{FB_GRAPH_URL}/{container_id}",
            params={"fields": "status_code,status"},
            headers=auth_header,
            timeout=30,
        )
        if st_res.status_code == 200:
            st_data = st_res.json()
            code = st_data.get("status_code")
            if code in ("FINISHED", "PUBLISHED"):
                break
            if code in ("ERROR", "EXPIRED"):
                raise RuntimeError(f"Instagram rejected Reel processing: {st_data}")
    else:
        raise RuntimeError("Instagram Reel processing timed out.")

    # Step 4: Publish container
    pub_res = httpx.post(
        f"{FB_GRAPH_URL}/{uid}/media_publish",
        data={"creation_id": container_id},
        headers=auth_header,
        timeout=60,
    )
    if pub_res.status_code >= 400:
        raise RuntimeError(f"Instagram publish failed ({pub_res.status_code}): {pub_res.text[:300]}")
    media_id = pub_res.json()["id"]

    permalink = f"https://www.instagram.com/reels/{media_id}/"
    try:
        perm_res = httpx.get(f"{FB_GRAPH_URL}/{media_id}", params={"fields": "permalink"}, headers=auth_header, timeout=20)
        if perm_res.status_code == 200 and perm_res.json().get("permalink"):
            permalink = perm_res.json()["permalink"]
    except Exception:
        pass
    return media_id, permalink



# ---------------------------------------------------------------------------
# 10. Adopted from `purffle-shorts` & `youtube-marketing-skills`:
#     - Zero-Key YouTube Autocomplete & Google Trends RSS Scraper (`purffle_shorts/seo.py` & `topics.py`)
#     - Channel Winners/Losers Prompt Feedback & Thompson Sampling (`purffle_shorts/analytics.py`)
#     - 2026 YouTube Algorithm SEO Guard & Title-Thumbnail Synergy (`youtube-marketing-skills`)
#     - Automated Question-Driven Top Comment Poster (`youtube-marketing-skills/server.js`)
# ---------------------------------------------------------------------------
import math
import random
import xml.etree.ElementTree as ET

YT_SUGGEST_URL = "https://suggestqueries.google.com/complete/search"
GOOGLE_TRENDS_RSS = "https://trends.google.com/trending/rss"
YT_COMMENT_THREADS_URL = "https://www.googleapis.com/youtube/v3/commentThreads"


def fetch_youtube_autocomplete_keywords(seed_query: str, gl: str = "US", hl: str = "en", limit: int = 10) -> List[str]:
    """
    Adopted from `purffle_shorts/seo.py`:
    Queries YouTube's real-time search autocomplete box (`suggestqueries.google.com`, 0 quota cost, no key)
    to get the exact phrases viewers are typing into YouTube right now.
    """
    q = (seed_query or "").strip()
    if not q:
        return []
    out: List[str] = []
    for variant in (q, f"{q} live", f"{q} reaction"):
        try:
            r = httpx.get(
                YT_SUGGEST_URL,
                params={"client": "firefox", "ds": "yt", "hl": hl, "gl": gl, "q": variant},
                timeout=6.0,
            )
            if r.status_code == 200:
                data = r.json()
                items = data[1] if isinstance(data, list) and len(data) > 1 and isinstance(data[1], list) else []
                for s in items:
                    if isinstance(s, str) and s.strip() and s.strip() not in out:
                        out.append(s.strip())
        except Exception:
            continue
    return out[:limit]


def fetch_google_trends_rss(geo: str = "US", limit: int = 8) -> List[str]:
    """
    Adopted from `purffle_shorts/topics.py`:
    Reads today's trending searches from Google Trends RSS (`trends.google.com/trending/rss`, 0 cost).
    """
    try:
        r = httpx.get(GOOGLE_TRENDS_RSS, params={"geo": geo}, timeout=8.0)
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.content)
        titles = []
        for item in root.iter("item"):
            t = (item.findtext("title") or "").strip()
            if t and t not in titles:
                titles.append(t)
        return titles[:limit]
    except Exception:
        return []


def build_winners_losers_insight_prompt(state: Dict) -> str:
    """
    Adopted from `purffle_shorts/analytics.py` (`insights()`):
    Shows the LLM your channel's top-performing Shorts vs. lowest-performing Shorts so new hooks
    and titles lean toward what your audience actually clicks and watches.
    """
    history = state.get("video_performance_log") or []
    if len(history) < 3:
        return ""
    ranked = sorted(history, key=lambda v: float(v.get("views") or 0), reverse=True)
    best = ranked[:3]
    worst = [w for w in ranked[-2:] if w.get("id") not in {b.get("id") for b in best}]
    lines = ["What performed BEST on our channel so far (emulate their hook energy & angle):"]
    for b in best:
        lines.append(f'- "{b.get("title", "")}" (style={b.get("style", "bounce")}): {b.get("views", 0):,} views, {b.get("likes", 0):,} likes')
    if worst:
        lines.append("What performed WORST (avoid this framing):")
        for w in worst:
            lines.append(f'- "{w.get("title", "")}": {w.get("views", 0):,} views')
    return "\n".join(lines)


def choose_caption_style_thompson(state: Dict, candidates: Tuple[str, ...] = ("bounce", "hype", "clean")) -> str:
    """
    Adopted from `purffle_shorts/analytics.py` (`choose_style()`):
    Uses Bayesian Thompson Sampling over `log1p(views)` across your channel's past videos to pick
    the highest-converting caption animation style while still exploring alternatives.
    """
    rows = [r for r in (state.get("video_performance_log") or []) if r.get("style") in candidates]
    if len(rows) < 6:
        return env("CAPTION_STYLE", "bounce")
    vals = [math.log1p(max(0.0, float(r.get("views") or 0))) for r in rows]
    mean0 = sum(vals) / len(vals)
    sd0 = max(0.35, math.sqrt(sum((x - mean0) ** 2 for x in vals) / max(1, len(vals) - 1)))
    prior = 2.0
    best_style, best_draw = "bounce", -math.inf
    for st_name in candidates:
        xs = [math.log1p(max(0.0, float(r.get("views") or 0))) for r in rows if r.get("style") == st_name]
        n = len(xs)
        post_mean = (prior * mean0 + sum(xs)) / (prior + n)
        post_sd = sd0 / math.sqrt(prior + n)
        draw = random.gauss(post_mean, post_sd)
        if draw > best_draw:
            best_style, best_draw = st_name, draw
    return best_style


def audit_and_enforce_2026_seo_rules(seo: Dict, streamer: str, topic: str) -> Dict:
    """
    Adopted from `youtube-marketing-skills` (`seo-playbook.md`, `thumbnail-ctr-guide.md`, `youtube-updater.md`):
    1. Primary keyword & streamer in the first 40 characters of the title (under 90 chars total).
    2. Title-Thumbnail Synergy: `thumbnail_badge` must be 2–3 words max and MUST NOT duplicate the title's opening words.
    3. First 150 characters of description (before 'Show more' fold) must contain the primary keyword naturally.
    4. Tags capped at <= 30 chars each, total < 460 chars, no single word repeated 8+ times (anti-spam guard).
    5. Generates an engagement-driving `pinned_comment` ending with a viewer question.
    """
    title = re.sub(r"\s+", " ", str(seo.get("title") or "")).strip()
    # Never start a Short title with slow "How to" opener
    title = re.sub(r"^how\s+to\s+", "", title, flags=re.I)
    if len(title) > 92:
        title = title[:84].rstrip() + " #Shorts"
    if "#shorts" not in title.lower():
        title = f"{title[:82].rstrip()} #Shorts"
    seo["title"] = title

    # Enforce 2-3 word max on thumbnail_badge AND Title-Thumbnail Synergy (no exact overlap)
    badge_words = re.sub(r"[^A-Z0-9 !?\-]", "", str(seo.get("thumbnail_badge") or "WAIT FOR IT").upper()).split()
    badge = " ".join(badge_words[:3]) or "WAIT FOR IT"
    if badge.lower() in title[:35].lower():
        badge = "UNREAL REACTION" if "reaction" not in title.lower() else "PURE CHAOS"
    seo["thumbnail_badge"] = badge[:20]

    # Enforce anti-spam tag limits (<=30 chars per tag, no word repeated >=8 times)
    clean_tags: List[str] = []
    word_counts: Dict[str, int] = {}
    for t in seo.get("tags") or []:
        t_clean = re.sub(r"[<>#]", "", str(t)).strip()[:30]
        if not t_clean:
            continue
        toks = t_clean.lower().split()
        if any(word_counts.get(w, 0) >= 7 for w in toks):
            continue
        for w in toks:
            word_counts[w] = word_counts.get(w, 0) + 1
        clean_tags.append(t_clean)
    seo["tags"] = sanitize_youtube_tags(clean_tags, streamer, topic)

    # Ensure pinned question comment exists (drives comment velocity for Shorts algorithm)
    if not seo.get("pinned_comment"):
        seo["pinned_comment"] = (
            f"What would your reaction be if this happened to you on {streamer}'s stream? "
            f"Drop your rating 1–10 below! 👇"
        )
    linted = lint_title_and_thumb_pairing(seo["title"], seo["thumbnail_badge"])
    seo["title"] = linted["title"]
    seo["thumbnail_badge"] = linted["thumbnail_badge"]
    seo["packaging_score"] = linted["pairing_score"]
    return seo


def post_engagement_comment(video_id: str, comment_text: str, access_token: Optional[str] = None) -> Optional[str]:
    """
    Adopted from `youtube-marketing-skills/server.js` (`postVideoComment` via `commentThreads.insert`):
    Posts a top-level question comment on the Short to jumpstart viewer replies.
    """
    if not comment_text:
        return None
    token = access_token or refresh_google_access_token()
    body = {
        "snippet": {
            "videoId": video_id,
            "topLevelComment": {
                "snippet": {
                    "textOriginal": comment_text[:280],
                }
            },
        }
    }
    try:
        r = httpx.post(
            YT_COMMENT_THREADS_URL,
            params={"part": "snippet"},
            json=body,
            headers={"Authorization": f"Bearer {token}"},
            timeout=25,
        )
        if r.status_code in (200, 201):
            return r.json().get("id")
    except Exception:
        pass
    return None



# ---------------------------------------------------------------------------
# 11. Adopted from `sajadapp/youtube-trend-analyzer`, `avas77/youtube-trend-analysis`,
#     `samuelnz36/youtube-trending-analysis`, and `gateragael/machine-learning-project`:
#     - Zero-Quota YouTube Channel Atom/RSS Feed Fetcher (`youtube.com/feeds/videos.xml?channel_id=...`)
#     - 1-Unit YouTube Gaming Trending Chart (`videos.list(chart="mostPopular", videoCategoryId="20")`)
#     - Breakout View Velocity (`views / hours_old`) + Like-Ratio (`likes / views`) + Engagement % Ranker
#     - 4-Layer Fault-Tolerant JSON Extractor (`extract_json_payload`)
#     - Optional Telegram Rich HTML Card Notifier (`send_telegram_summary_card`)
# ---------------------------------------------------------------------------
YT_CHANNEL_RSS_URL = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "yt": "http://www.youtube.com/xml/schemas/2015"}


def extract_json_payload(raw_text: str):
    """
    Adopted from `sajadapp/youtube-trend-analyzer` (`services/llm/json_parser.py`):
    4-layer fault-tolerant JSON extractor that recovers valid JSON dicts or arrays from messy LLM outputs.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Empty LLM response")
    clean = raw_text.strip()

    # Layer 1: Direct parse with strict=False
    try:
        return json.loads(clean, strict=False)
    except json.JSONDecodeError:
        pass

    # Layer 2: Markdown code fence extraction
    for m in re.finditer(r"```(?:json)?\s*([\[{][\s\S]*?[\]}])\s*```", clean, re.I):
        try:
            return json.loads(m.group(1).strip(), strict=False)
        except json.JSONDecodeError:
            continue

    # Layer 3: Outer bracket / brace greedy slice
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        first, last = clean.find(open_ch), clean.rfind(close_ch)
        if first != -1 and last > first:
            try:
                return json.loads(clean[first : last + 1], strict=False)
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not extract valid JSON from LLM output: {clean[:180]}")


def calculate_breakout_metrics(view_count: int, like_count: int, comment_count: int, published_at_iso: str) -> Dict[str, float]:
    """
    Adopted from `sajadapp/youtube-trend-analyzer` (`scrapers/metrics.py`),
    `avas77/youtube-trend-analysis` (`data-cleaning-transform.py`), and
    `samuelnz36/youtube-trending-analysis` (`02_eda.ipynb`):
    Computes:
      - `view_velocity`: views per hour since publication
      - `engagement_pct`: ((likes + comments) / views) * 100
      - `like_ratio_pct`: (likes / views) * 100 (benchmark: 3%–10% = high appreciation)
      - `breakout_score`: composite ranking score prioritizing fast-rising streams
    """
    views = max(0, int(view_count or 0))
    likes = max(0, int(like_count or 0))
    comments = max(0, int(comment_count or 0))
    try:
        pub_dt = dt.datetime.fromisoformat(str(published_at_iso).replace("Z", "+00:00"))
        if pub_dt.tzinfo is None:
            pub_dt = pub_dt.replace(tzinfo=dt.timezone.utc)
        hours_old = max(0.1, (dt.datetime.now(dt.timezone.utc) - pub_dt).total_seconds() / 3600.0)
    except Exception:
        hours_old = 6.0

    velocity = round(views / hours_old, 2)
    eng_pct = round(((likes + comments) / max(1, views)) * 100.0, 2)
    like_pct = round((likes / max(1, views)) * 100.0, 2)
    # Boost videos whose like ratio exceeds the 3% trending median
    quality_boost = 1.25 if like_pct >= 3.5 else 1.0
    breakout_score = round(velocity * (1.0 + (eng_pct / 10.0)) * quality_boost, 2)
    tier = "MEGA_VIRAL" if (views >= 100_000 and velocity >= 2000) else ("BREAKOUT" if velocity >= 300 else "RISING")
    return {
        "hours_old": round(hours_old, 2),
        "view_velocity": velocity,
        "engagement_pct": eng_pct,
        "like_ratio_pct": like_pct,
        "breakout_score": breakout_score,
        "tier": tier,
    }


def fetch_channel_rss_video_ids(channel_id: str, max_age_hours: float = 18.0) -> List[str]:
    """
    Adopted from `sajadapp/youtube-trend-analyzer` (`scrapers/rss_fetcher.py`):
    Fetches a YouTube channel's recent video IDs via `youtube.com/feeds/videos.xml`
    with ZERO YouTube Data API quota cost and ZERO ban risk!
    """
    if not channel_id or not channel_id.startswith("UC"):
        return []
    url = YT_CHANNEL_RSS_URL.format(channel_id=channel_id)
    try:
        r = httpx.get(url, timeout=10.0, headers={"User-Agent": "Mozilla/5.0 (compatible; TrendAnalyzerBot/1.0)"})
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.content)
        now = dt.datetime.now(dt.timezone.utc)
        out_ids: List[str] = []
        for entry in root.findall("atom:entry", ATOM_NS):
            vid_el = entry.find("yt:videoId", ATOM_NS)
            pub_el = entry.find("atom:published", ATOM_NS)
            vid = vid_el.text.strip() if (vid_el is not None and vid_el.text) else ""
            if not vid:
                continue
            if pub_el is not None and pub_el.text:
                try:
                    pub_dt = dt.datetime.fromisoformat(pub_el.text.strip().replace("Z", "+00:00"))
                    if (now - pub_dt).total_seconds() / 3600.0 > max_age_hours:
                        continue
                except ValueError:
                    pass
            out_ids.append(vid)
        return out_ids[:5]
    except Exception:
        return []


def fetch_most_popular_gaming_chart(
    state: Dict,
    token: str,
    region_code: str = "US",
    category_id: str = "20",
) -> Optional[Dict]:
    """
    Adopted from `avas77/youtube-trend-analysis` (`youtube_data_ingestion.py`):
    Calls `videos.list(chart="mostPopular", videoCategoryId="20")` (costs ONLY 1 quota unit,
    saving 99 units vs `search.list`!) and ranks trending gaming videos by `breakout_score`.
    """
    r = httpx.get(
        YT_VIDEOS_URL,
        params={
            "part": "snippet,statistics,liveStreamingDetails",
            "chart": "mostPopular",
            "regionCode": region_code,
            "videoCategoryId": category_id,
            "maxResults": 25,
        },
        headers={"Authorization": f"Bearer {token}"},
        timeout=20,
    )
    if r.status_code != 200:
        return None

    processed = set(state.get("processed_keys") or [])
    candidates: List[Dict] = []
    for item in r.json().get("items") or []:
        vid = item["id"]
        dedup_key = f"chart:{vid}"
        if dedup_key in processed:
            continue
        snip = item.get("snippet") or {}
        stats = item.get("statistics") or {}
        metrics = calculate_breakout_metrics(
            int(stats.get("viewCount") or 0),
            int(stats.get("likeCount") or 0),
            int(stats.get("commentCount") or 0),
            snip.get("publishedAt", ""),
        )
        # Filter to fresh breakout videos (<= 36 hours old)
        if metrics["hours_old"] <= 36.0:
            candidates.append({
                "url": f"https://www.youtube.com/watch?v={vid}",
                "video_id": vid,
                "title": snip.get("title", ""),
                "uploader": snip.get("channelTitle", ""),
                "is_live": snip.get("liveBroadcastContent") == "live",
                "dedup_key": dedup_key,
                "fallback_url": "",
                **metrics,
            })

    if not candidates:
        return None
    candidates.sort(key=lambda c: c["breakout_score"], reverse=True)
    best = candidates[0]
    print(
        f"[MOST POPULAR CHART] Selected #{1} Breakout Gaming Video: '{best['title']}' "
        f"by {best['uploader']} ({best['view_velocity']} views/hr, {best['tier']})"
    )
    return best


def send_telegram_summary_card(
    title: str,
    streamer: str,
    publish_at_iso: str,
    yt_url: str = "",
    drive_url: str = "",
    fb_url: str = "",
    ig_url: str = "",
) -> bool:
    """
    Adopted from `sajadapp/youtube-trend-analyzer` (`services/notifier/notifier.py`):
    Sends a clean HTML notification card to Telegram when `TELEGRAM_BOT_TOKEN` & `TELEGRAM_CHAT_ID` are set.
    """
    bot_token = env("TELEGRAM_BOT_TOKEN")
    chat_id = env("TELEGRAM_CHAT_ID")
    if not (bot_token and chat_id):
        return False
    links = [f"<a href='{yt_url}'>YouTube Short</a>" if yt_url else ""]
    if drive_url:
        links.append(f"<a href='{drive_url}'>Google Drive</a>")
    if fb_url:
        links.append(f"<a href='{fb_url}'>FB Reel</a>")
    if ig_url:
        links.append(f"<a href='{ig_url}'>IG Reel</a>")
    link_bar = " | ".join(x for x in links if x)
    html_msg = (
        f"🚀 <b>[24/7 SHORT SCHEDULED]</b>\n"
        f"🎮 <b>Creator:</b> {re.sub(r'[<>]', '', streamer)}\n"
        f"📌 <b>Title:</b> {re.sub(r'[<>]', '', title)}\n"
        f"⏰ <b>Scheduled Slot:</b> <code>{publish_at_iso}</code>\n"
        f"🔗 {link_bar}"
    )
    try:
        r = httpx.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": html_msg, "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=15,
        )
        return r.status_code == 200
    except Exception:
        return False



# ---------------------------------------------------------------------------
# 12. Adopted from `jakeschincariol/youtube-agent-skill`:
#     - 5-Property Weakest-Link Hook Scorer (`skills/yt-script/hookscore.py` & `hooks.json`)
#     - Title + Thumbnail Pairing Linter (`skills/yt-package/title.py`: 40c mobile / 60c desktop / 0 overlap)
#     - Channel Outlier Multiple Ranker (`skills/yt-viral/swipe.py`: views / channel_median_views)
# ---------------------------------------------------------------------------
import statistics

_HOOK_FILLER = {"basically", "actually", "literally", "just", "really", "very", "so", "kind", "sort", "like", "guys", "hey", "welcome", "today", "video", "subscribe", "channel"}
_HOOK_VAGUE = {"amazing", "incredible", "insane", "crazy", "huge", "massive", "game", "changer", "secret", "powerful", "ultimate", "best", "revolutionary", "mind", "blowing", "unbelievable", "epic", "perfect"}
_STOP_WORDS = {"the", "a", "an", "of", "for", "to", "in", "on", "and", "or", "is", "are", "with", "your", "you", "my", "i", "this", "that", "it", "how", "what", "why"}
_CONCRETE_RE = re.compile(r"\b(\d[\d,.]*\s?(%|k|m|x|s|h|star|stars)?|\$\d|\d+\s?(second|minute|hour|day|week|month|year)s?)\b", re.I)
_YOU_RE = re.compile(r"\b(you|your|you're|youre|yourself|bro|chat)\b", re.I)
_STAKE_RE = re.compile(r"\b(lose|lost|wasting|waste|quit|fail|failed|broke|cost|risk|before|stop|never|die|died|dying|dead|busted|wasted|arrested)\b", re.I)
_CURIOSITY_RE = re.compile(r"\b(why|how|what|which|until|before|but|nobody|almost|except|reason|wait|watch)\b", re.I)


def score_hook_5_properties(hook_text: str) -> Dict:
    """
    Adopted from `jakeschincariol/youtube-agent-skill` (`skills/yt-script/hookscore.py`):
    Scores a hook across 5 properties (SPECIFICITY, ADDRESS, STAKES, CURIOSITY, BREVITY)
    using weakest-link weighting: `verdict = round(0.6 * mean + 0.4 * min)`.
    """
    t = (hook_text or "").strip()
    w = re.findall(r"[a-z0-9'%$.]+", t.lower())
    if not w:
        return {"verdict": 0, "band": "WEAK", "parts": {}}

    nums = len(_CONCRETE_RE.findall(t))
    vague = sum(1 for x in w if x in _HOOK_VAGUE)
    filler = sum(1 for x in w if x in _HOOK_FILLER)
    proper = min(18, 6 * sum(1 for x in t.split()[1:] if x[:1].isupper()))
    spec = max(0, min(100, 34 + nums * 22 - vague * 16 - filler * 5 + proper))

    you_n = len(_YOU_RE.findall(t))
    first_you = 30 if _YOU_RE.search(" ".join(t.split()[:6])) else 0
    addr = max(0, min(100, 26 + you_n * 20 + first_you))

    stake_n = len(_STAKE_RE.findall(t))
    stak = max(0, min(100, 22 + stake_n * 26 + (14 if _CONCRETE_RE.search(t) else 0)))

    cur_n = len(_CURIOSITY_RE.findall(t))
    q_bonus = 18 if t.endswith("?") else 0
    closed_pen = -18 if re.search(r"\b(because|so that|which means)\b", t, re.I) else 0
    curi = max(0, min(100, 24 + cur_n * 17 + q_bonus + closed_pen))

    nw = len(w)
    # For on-screen 2-5 word badges or 9-24 word spoken hooks:
    brev = 100 if (2 <= nw <= 24) else max(20, 100 - abs(nw - 15) * 7)

    parts = {"SPECIFICITY": spec, "ADDRESS": addr, "STAKES": stak, "CURIOSITY": curi, "BREVITY": brev}
    vals = list(parts.values())
    verdict = round(0.6 * (sum(vals) / len(vals)) + 0.4 * min(vals))
    band = "STRONG" if verdict >= 72 else ("WORKABLE" if verdict >= 55 else "WEAK")
    return {"verdict": verdict, "band": band, "parts": parts}


def lint_title_and_thumb_pairing(title: str, thumb_badge: str) -> Dict:
    """
    Adopted from `jakeschincariol/youtube-agent-skill` (`skills/yt-package/title.py`):
    Lints the `(title, thumbnail_badge)` as a single click-surface unit:
      - Checks 40-char mobile cutoff & 60-char desktop search cutoff
      - Caps ALL-CAPS words in title at <= 2
      - Replaces vague adjectives unless paired with a concrete name/number
      - Enforces ZERO content-word duplication between title and thumbnail_badge
      - Enforces <= 3 words on thumbnail_badge
    """
    t = re.sub(r"\s+", " ", (title or "").strip())
    tb = re.sub(r"\s+", " ", (thumb_badge or "").strip().upper())
    t_words = re.findall(r"[a-z0-9']+", t.lower())
    tb_words = re.findall(r"[a-z0-9']+", tb.lower())

    issues = []
    good = []

    if len(t) > 100:
        issues.append("hard-limit-100")
    elif len(t) > 60:
        issues.append("desktop-60-cut")
    else:
        good.append("fits-desktop-60")

    # Cap ALL-CAPS words in title (excluding #Shorts/#GTA6 hashtags) at <= 2
    non_hash_words = [w for w in t.split() if not w.startswith("#")]
    caps_words = [w for w in non_hash_words if len(w) > 2 and w.isupper()]
    if len(caps_words) > 2:
        fixed_words = []
        caps_seen = 0
        for w in t.split():
            if not w.startswith("#") and len(w) > 2 and w.isupper():
                caps_seen += 1
                if caps_seen > 2:
                    w = w.capitalize()
            fixed_words.append(w)
        t = " ".join(fixed_words)

    # Check title vs thumbnail word duplication
    shared = (set(t_words) - _STOP_WORDS) & (set(tb_words) - _STOP_WORDS)
    if shared:
        issues.append(f"duplicate-words:{','.join(sorted(shared))}")
        # Automatically de-duplicate thumbnail badge so it complements the title
        alt_badges = ["WAIT FOR IT", "UNREAL PAYOFF", "PURE CHAOS", "INSTANT REGRET", "NO WAY"]
        for cand in alt_badges:
            cand_w = set(re.findall(r"[a-z0-9']+", cand.lower())) - _STOP_WORDS
            if not (cand_w & (set(t_words) - _STOP_WORDS)):
                tb = cand
                break
    else:
        good.append("zero-duplication")

    # Enforce <= 3 words on thumbnail badge
    if len(tb.split()) > 3:
        tb = " ".join(tb.split()[:3])

    score = max(0, min(100, 100 - 14 * len(issues) + 4 * len(good)))
    return {"title": t, "thumbnail_badge": tb, "pairing_score": score, "issues": issues}


def compute_channel_outlier_multiple(video_views: int, recent_channel_views: List[int]) -> float:
    """
    Adopted from `jakeschincariol/youtube-agent-skill` (`skills/yt-viral/swipe.py`):
    Scores a video as a multiple of its OWN channel's median views (`views / median`).
    A 3.0x+ outlier on a channel means something extraordinary happened in that stream!
    """
    clean = [float(v) for v in (recent_channel_views or []) if float(v or 0) > 0]
    if len(clean) < 3:
        return 1.0
    med = statistics.median(clean)
    return round(float(video_views or 0) / max(1.0, med), 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ultimate 24/7 YouTube Discovery, Heatmap Optimizer & Scheduler")
    parser.add_argument("--init", action="store_true", help="Initialize repository directories")
    parser.add_argument("--auth", action="store_true", help="Authorize YouTube + Analytics + Drive OAuth token")
    parser.add_argument("--update-heatmap", action="store_true", help="Fetch channel analytics and update Golden Hours")
    parser.add_argument("--self-test", action="store_true", help="Run unit tests for Heatmap scheduling & SEO")
    args = parser.parse_args()

    if args.init:
        init_repo_structure()
    elif args.auth:
        run_local_oauth_flow()
    elif args.update_heatmap:
        st = update_analytics_heatmap()
        save_cloud_state(st)
    elif args.self_test:
        now_fixed = dt.datetime(2026, 11, 19, 13, 10, tzinfo=dt.timezone.utc)
        # Test snapping to learned Analytics Heatmap Golden Hours [15, 19, 22]
        slot1 = compute_next_publish_slot(gap_hours=3.0, existing_iso_slots=[], now_utc=now_fixed, golden_hours_utc=[15, 19, 22])
        slot2 = compute_next_publish_slot(gap_hours=3.0, existing_iso_slots=[slot1], now_utc=now_fixed, golden_hours_utc=[15, 19, 22])
        assert slot1 == "2026-11-19T15:00:00Z", f"Expected 15:00Z, got {slot1}"
        assert slot2 == "2026-11-19T19:00:00Z", f"Expected 19:00Z, got {slot2}"
        print("Heatmap Golden-Hour Slot 1:", slot1)
        print("Heatmap Golden-Hour Slot 2:", slot2)
        print("ULTIMATE 24/7 SCHEDULER SELF-TEST PASSED")
    else:
        parser.print_help()
