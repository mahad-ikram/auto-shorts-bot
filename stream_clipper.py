"""
Step 2 (Upgraded) — Stateless Stream/Drive Clipper, ASR Transcriber, 9:16 ASS Renderer & Google Drive Sync
==========================================================================================================
Reverse-engineered and adapted for 100% online cloud execution (GitHub Actions + Google Drive) from
`haziqahmedqasmi/peakcut`.

What this module does:
1. Ingests video from EITHER:
   - Your **Google Drive Inbox folder** (`--from-drive` or `GDRIVE_INBOX_FOLDER_ID`) — bypassing YouTube
     datacenter IP blocks 100% in the cloud, OR
   - A live stream / VOD URL (`--url`, e.g., Twitch, Kick, or YouTube).
2. Transcribes audio with word-level timestamps using YouTube `json3` captions, Groq Cloud Whisper
   (`whisper-large-v3-turbo`), or local `sherpa-onnx` NVIDIA Parakeet 0.6B.
3. Computes per-second RMS loudness Z-scores (`(LOUD)` markers) and asks Gemini 2.5 Flash to pick
   the highest-retention 20–55s GTA VI moments.
4. Renders 1080x1920 vertical Shorts (`blur`, `zoom`, or `crop` layout) with custom TrueType-measured
   `.ass` word-pop captions (`bounce`, `hype`, `clean`) and `-14 LUFS` normalized audio.
5. Automatically saves finished `.mp4` Shorts, `_thumb.jpg` 3:2 thumbnails, and `_seo.json` sidecars
   back to your **Google Drive Output folder** (`--save-to-drive` or `GDRIVE_OUTPUT_FOLDER_ID`) AND
   schedules them on YouTube Studio (`--schedule`).
"""

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
import time
import urllib.request
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
import numpy as np

from youtube_seo_scheduler import (
    build_shorts_thumbnail,
    call_gemini_json_with_cascade,
    research_daily_viral_trends,
    compute_next_publish_slot,
    discover_active_stream,
    drive_download_file,
    drive_list_inbox_videos,
    drive_mark_processed,
    drive_upload_file,
    env,
    generate_gta6_seo_metadata,
    load_cloud_state,
    parse_drive_filename_metadata,
    refresh_google_access_token,
    save_cloud_state,
    update_analytics_heatmap,
    build_social_caption,
    upload_and_schedule_short,
    upload_facebook_reel,
    post_engagement_comment,
    send_telegram_summary_card,
    upload_instagram_reel,
)

# ---------------------------------------------------------------------------
# 1. Constants, Caption Styles & Roman Hinglish Transliterator
# ---------------------------------------------------------------------------
OUT_W, OUT_H, OUT_FPS = 1080, 1920, 30
CAP_MAX_W = 960
SR = 16000

STYLES = {
    "bounce": {
        "font": "Poppins ExtraBold",
        "size": 124,
        "base": "FFFFFF",
        "active": "00F0FF",
        "strong": "FFFFFF",
        "outline": 7,
        "pop": 112,
        "upper": True,
        "per": 3,
    },
    "hype": {
        "font": "Anton",
        "size": 140,
        "base": "FFFFFF",
        "active": "3C3CFF",
        "strong": "00D7FF",
        "outline": 7,
        "pop": 114,
        "upper": True,
        "per": 2,
    },
    "clean": {
        "font": "Poppins ExtraBold",
        "size": 96,
        "base": "FFFFFF",
        "active": "6BD7FF",
        "strong": "FFFFFF",
        "outline": 5,
        "pop": 106,
        "upper": False,
        "per": 4,
    },
}

HOOK_BOX = {
    "size": 88,
    "ink": "14100E",
    "box": "FFFFFF",
    "pad_x": 36,
    "pad_y": 22,
    "radius": 26,
    "max_w": 900,
}

_COMMON_ROMAN = {
    "में": "mein", "मैं": "main", "हैं": "hain", "है": "hai", "नहीं": "nahi",
    "क्या": "kya", "क्यों": "kyun", "यह": "yeh", "वह": "woh", "और": "aur",
    "भाई": "bhai", "यार": "yaar", "अरे": "arey", "देखो": "dekho",
    "یار": "yaar", "بھائی": "bhai", "کیا": "kya", "ہے": "hai", "نہیں": "nahi",
}


def romanize_text(text: str) -> str:
    try:
        import roman
        if roman.needs_roman(text or ""):
            return roman.to_roman(text or "")
    except Exception:
        pass
    out = []
    for tok in (text or "").split():
        clean = re.sub(r"[.,!?]+$", "", tok)
        punct = tok[len(clean):]
        out.append(_COMMON_ROMAN.get(clean, clean) + punct)
    return " ".join(out)


def run_cmd(cmd: List[str], cwd: Optional[Path] = None, timeout: int = 1800) -> str:
    p = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if p.returncode != 0:
        raise RuntimeError(f"{Path(str(cmd[0])).name} failed: {p.stderr[-1200:]}")
    return p.stdout


# ---------------------------------------------------------------------------
# 2. Google Drive Inbox Ingestion & Live/VOD Stream Ingestion
# ---------------------------------------------------------------------------
def ingest_from_google_drive(work_dir: Path, access_token: Optional[str] = None) -> Optional[Dict]:
    """
    Checks `GDRIVE_INBOX_FOLDER_ID` for the oldest unclipped video, downloads it into `work_dir`,
    and returns the standard ingestion dictionary (including `drive_file_id` and `drive_file_name`).
    """
    token = access_token or refresh_google_access_token()
    files = drive_list_inbox_videos(access_token=token)
    if not files:
        print("[INFO] No unclipped videos found in Google Drive Inbox folder.")
        return None

    target = files[0]
    fid, fname = target["id"], target["name"]
    ext = Path(fname).suffix or ".mp4"
    local_mp4 = work_dir / f"drive_source{ext}"
    print(f"[DRIVE INBOX] Downloading '{fname}' (ID: {fid}) from Google Drive...")
    drive_download_file(fid, local_mp4, access_token=token)

    streamer, stream_title = parse_drive_filename_metadata(fname)
    probe = probe_video(local_mp4)
    return {
        "video_path": local_mp4,
        "title": stream_title,
        "uploader": streamer,
        "is_live": False,
        "width": probe["width"],
        "height": probe["height"],
        "duration": probe["duration"],
        "native_words": None,
        "drive_file_id": fid,
        "drive_file_name": fname,
    }



# ---------------------------------------------------------------------------
# 2b. 10-Second Live Chat Explosion Spike Detector + 22-Second Pre-Roll Lookback
#     + Dead-Air / Filler Trimmer (`jakeschincariol/youtube-agent-skill/skills/yt-edit/deadair.py`)
# ---------------------------------------------------------------------------
import socket
import ssl
import threading

CHAT_BUCKET_SEC = 10          # Count chat messages in 10-second windows
PRE_SPIKE_ROLLBACK_SEC = 22.0 # Start the clip 22s BEFORE the chat spike (screen happens first, chat reacts after!)
POST_SPIKE_TAIL_SEC = 16.0    # Keep 16s after the spike peak to capture the streamer + chat aftermath
_FILLER_ONLY_RE = re.compile(
    r"^[\s,.-]*((um+|uh+|er+|ah+|so|okay|ok|right|yeah|like|anyway|basically|actually|you know|i mean|hold on)[\s,.-]*)+$",
    re.I,
)


def _record_twitch_irc_buckets(channel_login: str, duration_sec: int, stop_event: threading.Event, buckets: Dict[int, int]) -> None:
    """
    Connects anonymously to `irc.chat.twitch.tv:6697` during a live Twitch stream capture
    and counts incoming chat messages in 10-second buckets (`int(elapsed // 10)`).
    """
    t0 = time.time()
    try:
        raw = socket.create_connection(("irc.chat.twitch.tv", 6697), timeout=15)
        sock = ssl.create_default_context().wrap_socket(raw, server_hostname="irc.chat.twitch.tv")
        sock.sendall(f"NICK justinfan{int(t0) % 89999 + 10000}\r\nJOIN #{channel_login.lower()}\r\n".encode("utf-8"))
        sock.settimeout(5.0)
        buf = b""
        while not stop_event.is_set() and (time.time() - t0) < duration_sec:
            try:
                chunk = sock.recv(8192)
            except socket.timeout:
                continue
            if not chunk:
                break
            *lines, buf = (buf + chunk).split(b"\r\n")
            elapsed = max(0.0, time.time() - t0)
            b_idx = int(elapsed // CHAT_BUCKET_SEC)
            for ln in lines:
                if ln.startswith(b"PING"):
                    sock.sendall(b"PONG :tmi.twitch.tv\r\n")
                elif b" PRIVMSG #" in ln:
                    buckets[b_idx] = buckets.get(b_idx, 0) + 1
        sock.close()
    except Exception:
        pass


def parse_yt_live_chat_replay_buckets(work_dir: Path) -> Dict[int, int]:
    """
    Parses `yt-dlp` downloaded `.live_chat.json` replay files (if present) into 10-second message count buckets.
    """
    buckets: Dict[int, int] = {}
    for chat_file in work_dir.glob("*live_chat*.json*"):
        try:
            for line in chat_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                offset_ms = (
                    obj.get("videoOffsetTimeMsec")
                    or (obj.get("replayChatItemAction") or {}).get("videoOffsetTimeMsec")
                )
                if offset_ms is not None:
                    sec = max(0.0, float(offset_ms) / 1000.0)
                    b_idx = int(sec // CHAT_BUCKET_SEC)
                    buckets[b_idx] = buckets.get(b_idx, 0) + 1
        except Exception:
            continue
    return buckets


def detect_10s_chat_explosion_spikes(
    chat_buckets: Dict[int, int],
    total_duration_sec: float,
    pre_roll_sec: float = PRE_SPIKE_ROLLBACK_SEC,
    post_tail_sec: float = POST_SPIKE_TAIL_SEC,
) -> List[Dict]:
    """
    Analyzes 10-second chat message buckets (`chat_buckets[bucket_idx] = msg_count`).
    When a 10s bucket explodes (>= 2.5x rolling median or Z-score >= 2.2), it computes:
      - `spike_sec`: exact timestamp where chat exploded (`bucket_idx * 10`)
      - `clip_start`: `max(0.0, spike_sec - pre_roll_sec)` (22s BEFORE the chat spike, when the visual happened on screen!)
      - `clip_end`: `min(total_duration_sec, spike_sec + post_tail_sec)`
    """
    if not chat_buckets:
        return []
    max_b = max(int(total_duration_sec // CHAT_BUCKET_SEC), max(chat_buckets.keys(), default=0))
    counts = np.array([float(chat_buckets.get(i, 0)) for i in range(max_b + 1)], dtype=np.float32)
    if len(counts) < 3 or float(counts.max()) < 5.0:
        return []

    non_zero = counts[counts > 0]
    med = float(np.median(non_zero)) if len(non_zero) else 1.0
    mad = float(np.median(np.abs(non_zero - med))) * 1.4826 if len(non_zero) > 2 else max(1.0, med * 0.5)
    mad = max(1.0, mad)

    spikes: List[Dict] = []
    last_spike_sec = -999.0
    for b_idx, msg_count in enumerate(counts):
        z = (msg_count - med) / mad
        ratio = msg_count / max(1.0, med)
        spike_sec = float(b_idx * CHAT_BUCKET_SEC)
        if (z >= 2.2 or ratio >= 2.5) and msg_count >= 8 and (spike_sec - last_spike_sec) >= 35.0:
            clip_start = round(max(0.0, spike_sec - pre_roll_sec), 2)
            clip_end = round(min(total_duration_sec, spike_sec + post_tail_sec), 2)
            if clip_end - clip_start < 20.0:
                clip_end = round(min(total_duration_sec, clip_start + 32.0), 2)
            spikes.append({
                "spike_sec": spike_sec,
                "clip_start": clip_start,
                "clip_end": clip_end,
                "msgs_in_10s": int(msg_count),
                "jump_multiple": round(ratio, 2),
                "z_score": round(float(z), 2),
            })
            last_spike_sec = spike_sec

    spikes.sort(key=lambda s: (s["msgs_in_10s"], s["jump_multiple"]), reverse=True)
    return spikes


def trim_dead_air_and_fillers(words: List[Dict], start: float, end: float) -> Tuple[float, float]:
    """
    Adopted from `jakeschincariol/youtube-agent-skill` (`skills/yt-edit/deadair.py`):
    Trims leading/trailing filler-only words ("um", "uh", "hold on") and dead-air gaps > 0.6s
    at the edges of the clip so the first frame hooks immediately.
    """
    inside = [w for w in words if start <= w["s"] <= end and str(w.get("w", "")).strip()]
    while len(inside) > 3 and _FILLER_ONLY_RE.match(inside[0]["w"]):
        inside.pop(0)
    while len(inside) > 3 and _FILLER_ONLY_RE.match(inside[-1]["w"]):
        inside.pop()
    if not inside:
        return start, end
    new_s = max(start, round(inside[0]["s"] - 0.15, 2))
    new_e = min(end, round(inside[-1]["e"] + 0.25, 2))
    return (new_s, new_e) if (new_e - new_s) >= 4.0 else (start, end)


def _ydl_base_opts() -> Dict:
    opts: Dict = {
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 25,
        "concurrent_fragment_downloads": 4,
        "extractor_args": {"youtube": {"player_client": ["ios", "android_vr", "tv_embedded", "mweb", "web"]}},
    }
    cookie_file = env("YT_COOKIES_FILE")
    if cookie_file and Path(cookie_file).is_file():
        opts["cookiefile"] = cookie_file
    proxy = env("YT_PROXY")
    if proxy:
        opts["proxy"] = proxy
    return opts


def capture_stream_slice(
    url: str,
    work_dir: Path,
    slice_minutes: float = 15.0,
    max_height: int = 720,
) -> Dict:
    """Captures a bounded live stream slice or VOD via `yt-dlp` + `ffmpeg`."""
    import yt_dlp

    work_dir.mkdir(parents=True, exist_ok=True)
    src_mp4 = work_dir / "source.mp4"

    with yt_dlp.YoutubeDL(_ydl_base_opts()) as ydl:
        info = ydl.extract_info(url, download=False)

    is_live = bool(info.get("is_live"))
    title = info.get("title") or "GTA 6 Live Stream"
    uploader = info.get("uploader") or info.get("channel") or "Streamer"

    if is_live:
        fmt_opts = {**_ydl_base_opts(), "format": f"b[height<={max_height}]/bv*[height<={max_height}]+ba/b"}
        with yt_dlp.YoutubeDL(fmt_opts) as ydl:
            live_info = ydl.extract_info(url, download=False)
        fmts = live_info.get("requested_formats") or [live_info]
        stream_urls = [f["url"] for f in fmts if f.get("url")]

        dur_sec = max(30, int(slice_minutes * 60))
        chat_buckets: Dict[int, int] = {}
        stop_chat = threading.Event()
        m_tw = re.match(r"^https?://(?:www\.)?twitch\.tv/([A-Za-z0-9_]+)", url)
        chat_thread = None
        if m_tw:
            chat_thread = threading.Thread(
                target=_record_twitch_irc_buckets,
                args=(m_tw.group(1), dur_sec, stop_chat, chat_buckets),
                daemon=True,
            )
            chat_thread.start()

        cmd = ["ffmpeg", "-y", "-v", "error", "-reconnect", "1", "-reconnect_streamed", "1"]
        for u in stream_urls:
            cmd += ["-i", u]
        a_map = "1:a:0" if len(stream_urls) > 1 else "0:a:0"
        cmd += [
            "-map", "0:v:0",
            "-map", a_map,
            "-t", str(dur_sec),
            "-c:v", "copy",
            "-c:a", "aac",
            "-movflags", "+faststart",
            str(src_mp4),
        ]
        try:
            run_cmd(cmd, timeout=dur_sec + 180)
        finally:
            stop_chat.set()
            if chat_thread:
                chat_thread.join(timeout=3.0)
        native_words = None
    else:
        dl_opts = {
            **_ydl_base_opts(),
            "format": f"bv*[height<={max_height}][vcodec^=avc1]+ba[ext=m4a]/b[height<={max_height}]/b",
            "merge_output_format": "mp4",
            "outtmpl": str(work_dir / "source.%(ext)s"),
        }
        with yt_dlp.YoutubeDL(dl_opts) as ydl:
            ydl.download([url])
        candidates = sorted(work_dir.glob("source.*"), key=lambda p: p.stat().st_size, reverse=True)
        if not candidates:
            raise RuntimeError("VOD download completed but produced no media file.")
        src_mp4 = candidates[0]
        native_words = try_youtube_json3_words(info)
        chat_buckets = parse_yt_live_chat_replay_buckets(work_dir)

    probe = probe_video(src_mp4)
    return {
        "video_path": src_mp4,
        "title": title,
        "uploader": uploader,
        "is_live": is_live,
        "width": probe["width"],
        "height": probe["height"],
        "duration": probe["duration"],
        "native_words": native_words,
        "chat_buckets": chat_buckets,
        "drive_file_id": None,
        "drive_file_name": None,
    }


def try_youtube_json3_words(info: Dict) -> Optional[List[Dict]]:
    auto = info.get("automatic_captions") or {}
    key = next((k for k in auto if k.endswith("-orig")), None) or ("en" if "en" in auto else None)
    if not key:
        return None
    track = next((f for f in auto.get(key) or [] if f.get("ext") == "json3" and f.get("url")), None)
    if not track:
        return None
    try:
        raw = urllib.request.urlopen(track["url"], timeout=15).read(25_000_000)
        data = json.loads(raw)
        out: List[Dict] = []
        for ev in data.get("events") or []:
            segs = [x for x in ev.get("segs") or [] if (x.get("utf8") or "").strip()]
            if not segs:
                continue
            t0, dur = float(ev.get("tStartMs") or 0), float(ev.get("dDurationMs") or 0)
            timed = len(segs) > 1 or "tOffsetMs" in segs[0]
            for x in segs:
                parts = x["utf8"].split()
                for idx, w in enumerate(parts):
                    if re.fullmatch(r"[\[(].*[\])]", w):
                        continue
                    at = t0 + float(x.get("tOffsetMs") or 0) if timed else t0 + dur * idx / max(1, len(parts))
                    out.append({"w": romanize_text(w), "s": round(at / 1000.0, 2)})
        out.sort(key=lambda item: item["s"])
        for i, w in enumerate(out):
            nxt = out[i + 1]["s"] if i + 1 < len(out) else w["s"] + 0.4
            w["e"] = round(max(w["s"] + 0.08, min(nxt, w["s"] + 0.6)), 2)
        return out or None
    except Exception:
        return None


def probe_video(path: Path) -> Dict:
    out = run_cmd(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)])
    data = json.loads(out)
    v = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if not v:
        raise RuntimeError(f"No video stream found in {path}")
    return {
        "width": int(v["width"]),
        "height": int(v["height"]),
        "duration": float(data.get("format", {}).get("duration", 0.0)),
    }


def extract_16k_wav(video_path: Path, wav_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Extract 16 kHz mono PCM WAV using standard library `wave` and compute RMS dB per second."""
    run_cmd(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", str(SR),
            "-af", "aresample=async=1:first_pts=0",
            "-c:a", "pcm_s16le",
            str(wav_path),
        ]
    )
    with wave.open(str(wav_path), "rb") as wf:
        assert wf.getframerate() == SR
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    n_sec = len(pcm) // SR
    if n_sec == 0:
        return pcm, np.zeros(1, dtype=np.float32)
    frames = pcm[: n_sec * SR].reshape(n_sec, SR).astype(np.float32) / 32768.0
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-10)
    db_per_sec = (20.0 * np.log10(rms)).astype(np.float32)
    return pcm, db_per_sec


# ---------------------------------------------------------------------------
# 3. Speech-to-Text Transcriber (Groq Whisper Fast-Path + Sherpa-ONNX Parakeet)
# ---------------------------------------------------------------------------
def transcribe_audio(wav_path: Path, pcm: np.ndarray) -> List[Dict]:
    groq_key = env("GROQ_API_KEY")
    if groq_key:
        try:
            with wav_path.open("rb") as f:
                r = httpx.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {groq_key}"},
                    files={"file": (wav_path.name, f, "audio/wav")},
                    data={
                        "model": "whisper-large-v3-turbo",
                        "response_format": "verbose_json",
                        "timestamp_granularities[]": "word",
                    },
                    timeout=120,
                )
            r.raise_for_status()
            words = r.json().get("words") or []
            if words:
                return [
                    {
                        "w": romanize_text(w.get("word", "").strip()),
                        "s": round(float(w.get("start", 0.0)), 2),
                        "e": round(float(w.get("end", 0.0)), 2),
                    }
                    for w in words
                    if w.get("word", "").strip()
                ]
        except Exception as exc:
            print(f"[WARN] Groq Whisper API failed, falling back to local sherpa-onnx: {exc}")

    return _transcribe_sherpa_onnx(pcm)


def _transcribe_sherpa_onnx(pcm: np.ndarray) -> List[Dict]:
    import sherpa_onnx

    model_dir = Path(env("ASR_MODEL_DIR", "models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"))
    if not (model_dir / "tokens.txt").is_file():
        raise FileNotFoundError(
            f"Local ASR model not found in {model_dir}. Set GROQ_API_KEY for cloud Whisper "
            f"or let GitHub Actions cache download Parakeet 0.6B."
        )
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(model_dir / "encoder.int8.onnx"),
        decoder=str(model_dir / "decoder.int8.onnx"),
        joiner=str(model_dir / "joiner.int8.onnx"),
        tokens=str(model_dir / "tokens.txt"),
        num_threads=max(2, (os.cpu_count() or 4) - 1),
        sample_rate=SR,
        feature_dim=80,
        decoding_method="greedy_search",
        model_type="nemo_transducer",
    )
    samples = pcm.astype(np.float32) / 32768.0
    words: List[Dict] = []
    chunk_len = SR * 20
    for offset in range(0, len(samples), chunk_len):
        piece = samples[offset : offset + chunk_len]
        st = rec.create_stream()
        st.accept_waveform(SR, piece)
        rec.decode_stream(st)
        res = st.result
        base_sec = offset / SR
        cur_w, cur_s = "", 0.0
        for tok, ts in zip(res.tokens, res.timestamps):
            if tok.startswith(" ") or not cur_w:
                if cur_w.strip():
                    words.append({"w": romanize_text(cur_w.strip()), "s": round(cur_s, 2), "e": round(base_sec + ts, 2)})
                cur_w = tok
                cur_s = base_sec + ts
            else:
                cur_w += tok
        if cur_w.strip():
            words.append({"w": romanize_text(cur_w.strip()), "s": round(cur_s, 2), "e": round(cur_s + 0.35, 2)})
    return words


# ---------------------------------------------------------------------------
# 4. Loudness Z-Score & AI Viral Clip Scout
# ---------------------------------------------------------------------------
CLIP_SCOUT_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "start": {"type": "NUMBER"},
            "end": {"type": "NUMBER"},
            "score": {"type": "INTEGER"},
            "hook": {"type": "STRING"},
            "highlight_words": {"type": "ARRAY", "items": {"type": "STRING"}},
            "reason": {"type": "STRING"},
        },
        "required": ["start", "end", "score", "hook", "highlight_words", "reason"],
    },
}



def snap_clip_to_sentences(words: List[Dict], start: float, end: float, min_s: float = 18.0, max_s: float = 58.0) -> Tuple[float, float]:
    """
    Adopted from `purffle_shorts/clipper.py` (`_snap()`):
    Groups word timestamps into complete spoken sentences (split on `.!?` or pauses > 0.85s)
    and snaps `start` and `end` to sentence boundaries so streamers are never cut mid-word.
    """
    if not words:
        return start, end
    sents: List[Tuple[float, float]] = []
    cur_s = words[0]["s"]
    for i, w in enumerate(words):
        nxt = words[i + 1] if i + 1 < len(words) else None
        pause = (nxt["s"] - w["e"]) if nxt else 9.0
        if str(w["w"]).rstrip().endswith((".", "!", "?")) or pause > 0.85:
            sents.append((cur_s, w["e"]))
            if nxt:
                cur_s = nxt["s"]
    if not sents:
        return start, end
    i = min(range(len(sents)), key=lambda k: abs(sents[k][0] - start))
    j = min(range(i, len(sents)), key=lambda k: abs(sents[k][1] - end))
    while j > i and sents[j][1] - sents[i][0] > max_s:
        j -= 1
    while j + 1 < len(sents) and sents[j][1] - sents[i][0] < min_s:
        j += 1
    s0, e0 = sents[i][0], min(sents[j][1], sents[i][0] + max_s)
    return (round(s0, 2), round(e0, 2)) if (e0 - s0) >= 4.0 else (start, end)


def detect_face_center_x(video_path: Path, start: float, dur: float) -> Optional[float]:
    """
    Adopted from `purffle_shorts/clipper.py` (`_face_detector` & `plan_face_crop`):
    Samples frames from the clip using OpenCV's frontal face detector to find the median horizontal
    position (0.0..1.0) of the streamer's face so center-crops never slice off a corner webcam!
    """
    try:
        import cv2
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if not os.path.exists(cascade_path):
            return None
        det = cv2.CascadeClassifier(cascade_path)
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        xs: List[float] = []
        for offset in (0.5, dur * 0.35, dur * 0.65):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int((start + offset) * fps))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = det.detectMultiScale(gray, 1.1, 5, minSize=(28, 28))
            if len(faces):
                best = max(faces, key=lambda f: f[2] * f[3])
                xs.append((best[0] + best[2] / 2.0) / frame.shape[1])
        cap.release()
        if xs:
            xs.sort()
            return round(xs[len(xs) // 2], 4)
    except Exception:
        pass
    return None


def loud_zscore(db: np.ndarray) -> np.ndarray:
    if len(db) < 8:
        return np.zeros_like(db)
    valid = db[db > -65.0]
    if len(valid) < 5:
        return np.zeros_like(db)
    med = float(np.median(valid))
    mad = float(np.median(np.abs(valid - med))) * 1.4826 or 1.0
    return np.clip((db - med) / mad, -3.0, 6.0)


def pick_viral_gta6_moments(
    words: List[Dict],
    db: np.ndarray,
    title: str,
    uploader: str,
    max_clips: int = 2,
    min_len: int = 22,
    max_len: int = 55,
    chat_buckets: Optional[Dict[int, int]] = None,
) -> List[Dict]:
    z = loud_zscore(db)
    total_dur_est = float(len(db)) if len(db) else (words[-1]['e'] if words else 60.0)
    chat_spikes = detect_10s_chat_explosion_spikes(chat_buckets or {}, total_dur_est)
    chat_pre_roll_secs = {int(s['clip_start']): s for s in chat_spikes}
    lines: List[str] = []
    cur_line: List[str] = []
    line_start = 0.0

    for w in words:
        if not cur_line:
            line_start = w["s"]
        cur_line.append(w["w"])
        if len(cur_line) >= 12 or w["w"].endswith((".", "!", "?")):
            sec_idx = min(len(z) - 1, max(0, int(line_start))) if len(z) else 0
            loud_tag = " (LOUD)" if (len(z) and float(z[sec_idx : sec_idx + 4].max(initial=0.0)) >= 1.35) else ""
            chat_hit = next((cs for cs in chat_spikes if abs(line_start - cs["clip_start"]) <= 8.0), None)
            if chat_hit:
                loud_tag += f" (CHAT EXPLODED {chat_hit['msgs_in_10s']} MSGS/10s AT {chat_hit['spike_sec']:.0f}s -> START CLIP HERE 22s BEFORE SPIKE!)"
            lines.append(f"[{line_start:.1f}s]{loud_tag} " + " ".join(cur_line))
            cur_line = []
    if cur_line:
        lines.append(f"[{line_start:.1f}s] " + " ".join(cur_line))

    key = env("AI_API_KEY")
    if key and lines:
        prompt = (
            f"You are a viral YouTube Shorts scout (Topic/Niche: {env('TOPIC_NICHE', 'auto')}).\n"
            f"Streamer: {uploader} | Stream: {title}\n"
            f"Pick up to {max_clips} standalone clips ({min_len}s to {max_len}s each) with the strongest reaction, "
            f"funny fail, clutch play, hot take, or shocking moment. Lines marked (LOUD) indicate audio spikes.\n"
            f"Rules:\n"
            f"- hook: max 5 words ALL CAPS shown on video.\n"
            f"- highlight_words: 3 to 6 punch words spoken in the clip.\n\n"
            f"Transcript:\n" + "\n".join(lines[:450])
        )
        try:
            picks = call_gemini_json_with_cascade(prompt, CLIP_SCOUT_SCHEMA, ai_key=key)
            if isinstance(picks, list) and picks:
                return picks[:max_clips]
        except Exception as exc:
            print(f"[WARN] AI clip scout cascade fell back to loudness peak detector ({exc})")

    total_dur = float(len(db)) if len(db) else (words[-1]["e"] if words else 30.0)
    if chat_spikes:
        top_cs = chat_spikes[0]
        start_s, end_s = top_cs["clip_start"], top_cs["clip_end"]
    else:
        peak_sec = int(np.argmax(db)) if len(db) else int(total_dur // 2)
        start_s = max(0.0, float(peak_sec - 12))
        end_s = min(total_dur, start_s + min(max_len, max(min_len, 30)))
    if end_s - start_s < 5.0:
        start_s, end_s = 0.0, min(total_dur, 30.0)
    return [
        {
            "start": round(start_s, 2),
            "end": round(end_s, 2),
            "score": 85,
            "hook": "WAIT FOR THIS REACTION",
            "highlight_words": ["wait", "insane", "bro", "police", "vice"],
            "reason": "Peak audio loudness spike detected.",
        }
    ]


# ---------------------------------------------------------------------------
# 5. TrueType Binary Font Measurer & 9:16 ASS Renderer
# ---------------------------------------------------------------------------
def _find_system_ttf() -> Path:
    candidates = [
        Path("fonts/Poppins-ExtraBold.ttf"),
        Path("fonts/Anton-Regular.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"),
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError("No TrueType (.ttf) font found in fonts/ or /usr/share/fonts/")


class _FontMetrics:
    def __init__(self, ttf_path: Path):
        self.path = ttf_path
        from PIL import ImageFont

        self.pil_font = ImageFont.truetype(str(ttf_path), 200)
        raw = ttf_path.read_bytes()
        tables = {}
        for i in range(struct.unpack(">H", raw[4:6])[0]):
            tag, _, off, _ = struct.unpack(">4sIII", raw[12 + 16 * i : 28 + 16 * i])
            tables[tag] = off
        self.upm = struct.unpack(">H", raw[tables[b"head"] + 18 : tables[b"head"] + 20])[0]
        o = tables[b"OS/2"]
        self.asc, self.desc = struct.unpack(">HH", raw[o + 74 : o + 78])
        self.cap = struct.unpack(">h", raw[o + 88 : o + 90])[0] or int(self.upm * 0.7)

    def em(self, size: float) -> float:
        return size * self.upm / max(1, (self.asc + self.desc))

    def width(self, text: str, size: float) -> float:
        return self.pil_font.getlength(text) * self.em(size) / 200.0

    def cap_shift(self, size: float) -> float:
        return ((self.asc - self.desc) / 2.0 - self.cap / 2.0) * size / max(1, (self.asc + self.desc))


def _ass_ts(t: float) -> str:
    t = max(0.0, t)
    h, m = int(t // 3600), int(t % 3600 // 60)
    return f"{h}:{m:02d}:{t % 60:05.2f}"


def build_ass_subtitles(
    words: List[Dict],
    start: float,
    end: float,
    hook: str,
    style_name: str,
    highlight_words: List[str],
    hook_y: int,
    cap_y: int,
    ttf_path: Path,
) -> str:
    st = STYLES.get(style_name, STYLES["bounce"])
    fm = _FontMetrics(ttf_path)
    strong = {re.sub(r"[^a-z0-9']", "", h.lower()) for h in (highlight_words or [])}
    dur = max(0.5, end - start)

    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {OUT_W}
PlayResY: {OUT_H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{st['font']},{st['size']},&H00{st['base']},&H00{st['base']},&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,{st['outline']},0,5,0,0,0,1
Style: Hook,{st['font']},{HOOK_BOX['size']},&H00{HOOK_BOX['ink']},&H00{HOOK_BOX['ink']},&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: List[str] = []

    clean_hook = re.sub(r"[{}\\]", "", (hook or "").strip())
    if clean_hook:
        h_size = HOOK_BOX["size"]
        hw = fm.width(clean_hook, h_size)
        if hw > HOOK_BOX["max_w"]:
            h_size = max(44, int(h_size * HOOK_BOX["max_w"] / hw))
            hw = fm.width(clean_hook, h_size)
        line_h = fm.em(h_size) * 1.16
        bw, bh = hw + 2 * HOOK_BOX["pad_x"], line_h + 2 * HOOK_BOX["pad_y"]
        x0, y0 = (OUT_W - bw) / 2, hook_y - bh / 2
        x1, y1 = x0 + bw, y0 + bh
        r = min(HOOK_BOX["radius"], bh / 2)
        n = lambda v: f"{v:.0f}"
        box_draw = (
            f"m {n(x0 + r)} {n(y0)} l {n(x1 - r)} {n(y0)} "
            f"b {n(x1)} {n(y0)} {n(x1)} {n(y0)} {n(x1)} {n(y0 + r)} "
            f"l {n(x1)} {n(y1 - r)} b {n(x1)} {n(y1)} {n(x1)} {n(y1)} {n(x1 - r)} {n(y1)} "
            f"l {n(x0 + r)} {n(y1)} b {n(x0)} {n(y1)} {n(x0)} {n(y1)} {n(x0)} {n(y1 - r)} "
            f"l {n(x0)} {n(y0 + r)} b {n(x0)} {n(y0)} {n(x0)} {n(y0)} {n(x0 + r)} {n(y0)}"
        )
        span = f"{_ass_ts(0)},{_ass_ts(dur)}"
        events.append(f"Dialogue: 3,{span},Hook,,0,0,0,,{{\\an7\\pos(0,0)\\fad(150,0)\\c&H{HOOK_BOX['box']}&\\p1}}{box_draw}{{\\p0}}")
        cy = hook_y - fm.cap_shift(h_size)
        events.append(f"Dialogue: 4,{span},Hook,,0,0,0,,{{\\an5\\pos({OUT_W // 2},{cy:.0f})\\fad(150,0)\\fs{h_size}}}{clean_hook}")

    clip_words = [dict(w, w=re.sub(r"[{}\\]", "", str(w["w"])).strip()) for w in words if start - 0.05 <= w["s"] < end]
    clip_words = [w for w in clip_words if w["w"]]
    groups: List[List[Dict]] = []
    cur: List[Dict] = []
    for w in clip_words:
        if cur and (
            len(cur) >= st["per"]
            or sum(len(x["w"]) for x in cur) + len(w["w"]) > 16
            or w["s"] - cur[-1]["e"] > 0.45
        ):
            groups.append(cur)
            cur = []
        cur.append(w)
    if cur:
        groups.append(cur)

    for gi, g in enumerate(groups):
        g_start = max(0.0, g[0]["s"] - start)
        g_end = min(g[-1]["e"] + 0.25, groups[gi + 1][0]["s"] if gi + 1 < len(groups) else end) - start
        if g_end - g_start < 0.02:
            continue
        texts = [re.sub(r"[.,;:]+$", "", x["w"]).upper() if st["upper"] else re.sub(r"[.,;:]+$", "", x["w"]) for x in g]
        size = st["size"]
        gap = fm.width("n n", size) - 2 * fm.width("n", size) + (2 * st["outline"] + 6)
        total_w = sum(fm.width(t, size) for t in texts) + gap * (len(texts) - 1)
        if total_w > CAP_MAX_W:
            size = max(40, int(size * CAP_MAX_W / total_w))
            gap = fm.width("n n", size) - 2 * fm.width("n", size) + (2 * st["outline"] + 6)
            total_w = sum(fm.width(t, size) for t in texts) + gap * (len(texts) - 1)

        cy = cap_y - fm.cap_shift(size)
        x = (OUT_W - total_w) / 2.0
        for i, (w, t) in enumerate(zip(g, texts)):
            wd = fm.width(t, size)
            pop = max(104, min(st["pop"], 100 + int(2200 / max(wd, 1))))
            pos = f"\\an5\\pos({x + wd / 2:.0f},{cy:.0f})\\fs{size}"
            x += wd + gap
            a = max(g_start, w["s"] - start)
            b = (g[i + 1]["s"] - start) if i + 1 < len(g) else g_end
            w_key = re.sub(r"[^a-z0-9']", "", w["w"].lower())
            base_col = st["strong"] if w_key in strong else st["base"]

            if a - g_start >= 0.02:
                events.append(f"Dialogue: 1,{_ass_ts(g_start)},{_ass_ts(a)},Cap,,0,0,0,,{{{pos}\\c&H{base_col}&}}{t}")
            if b - a >= 0.02:
                events.append(
                    f"Dialogue: 2,{_ass_ts(a)},{_ass_ts(b)},Cap,,0,0,0,,{{{pos}\\c&H{st['active']}&\\fscx94\\fscy94"
                    f"\\t(0,90,\\fscx{pop}\\fscy{pop})\\t(90,200,\\fscx{pop - 5}\\fscy{pop - 5})}}{t}"
                )
            if g_end - b >= 0.02:
                events.append(
                    f"Dialogue: 1,{_ass_ts(b)},{_ass_ts(g_end)},Cap,,0,0,0,,{{{pos}\\c&H{base_col}&"
                    f"\\fscx{pop - 5}\\fscy{pop - 5}\\t(0,90,\\fscx100\\fscy100)}}{t}"
                )
    return head + "\n".join(events) + "\n"


def render_vertical_short(
    src_mp4: Path,
    out_dir: Path,
    stem: str,
    clip: Dict,
    words: List[Dict],
    src_w: int,
    src_h: int,
    layout: str = "blur",
    style_name: str = "bounce",
    isolate_voice_band: bool = False,
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    start, end = snap_clip_to_sentences(words, float(clip["start"]), float(clip["end"]))
    ttf_path = _find_system_ttf()

    if layout == "crop" or src_h >= src_w:
        fg_h, hook_y, cap_y = OUT_H, 300, 1380
        vf = f"[0:v]fps={OUT_FPS},scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,crop={OUT_W}:{OUT_H},setsar=1[v0];"
    else:
        fg_h = 1000 if layout == "zoom" else min(OUT_H, int(round(OUT_W * src_h / src_w / 2)) * 2)
        top = (OUT_H - fg_h) // 2
        hook_y, cap_y = max(230, top - 190), min(1640, top + fg_h + 180)
        vf = (
            f"[0:v]fps={OUT_FPS},split=2[a][b];"
            f"[a]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=6:2,"
            f"scale={OUT_W}:{OUT_H},eq=brightness=-0.14:saturation=1.15[bg];"
            f"[b]scale={OUT_W}:{fg_h}:force_original_aspect_ratio=increase,crop={OUT_W}:{fg_h}[fg];"
            f"[bg][fg]overlay=0:(H-h)/2,setsar=1[v0];"
        )

    ass_text = build_ass_subtitles(
        words=words,
        start=start,
        end=end,
        hook=clip.get("hook", "GTA 6 LIVE"),
        style_name=style_name,
        highlight_words=clip.get("highlight_words", []),
        hook_y=hook_y,
        cap_y=cap_y,
        ttf_path=ttf_path,
    )

    work = Path(tempfile.mkdtemp(prefix="gta6_pk_"))
    try:
        (work / "sub.ass").write_text(ass_text, encoding="utf-8")
        (work / "f").mkdir()
        shutil.copyfile(ttf_path, work / "f" / ttf_path.name)
        for extra_ttf in Path("fonts").glob("*.ttf"):
            shutil.copyfile(extra_ttf, work / "f" / extra_ttf.name)

        fc = vf + "[v0]ass=sub.ass:fontsdir=f[v]"
        af = "highpass=f=110,lowpass=f=7800,loudnorm=I=-14:TP=-1.5:LRA=11" if isolate_voice_band else "loudnorm=I=-14:TP=-1.5:LRA=11"

        run_cmd(
            [
                "ffmpeg", "-y", "-v", "error",
                "-ss", f"{start:.2f}",
                "-t", f"{max(1.0, end - start):.2f}",
                "-i", str(src_mp4.resolve()),
                "-filter_complex", fc,
                "-map", "[v]",
                "-map", "0:a:0",
                "-af", af,
                "-c:v", "libx264", "-crf", "21", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
                "-movflags", "+faststart",
                "out.mp4",
            ],
            cwd=work,
        )

        cover_t = min(1.2, max(0.2, (end - start) / 2.0))
        run_cmd(
            [
                "ffmpeg", "-y", "-v", "error",
                "-ss", f"{cover_t:.2f}",
                "-i", "out.mp4",
                "-frames:v", "1", "-q:v", "3",
                "raw_cover.jpg",
            ],
            cwd=work,
        )

        final_mp4 = out_dir / f"{stem}.mp4"
        final_thumb = out_dir / f"{stem}_thumb.jpg"
        shutil.move(str(work / "out.mp4"), str(final_mp4))
        build_shorts_thumbnail(
            frame_jpg=work / "raw_cover.jpg",
            out_jpg=final_thumb,
            badge_text=clip.get("hook", "GTA 6 LIVE"),
            font_path=ttf_path,
        )
        return {"mp4": final_mp4, "thumb": final_thumb}
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------------------
# 6. End-to-End Stateless CLI Entrypoint & Self-Test
# ---------------------------------------------------------------------------
def run_self_test() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="gta6_test_"))
    try:
        src_vid = tmp / "sample_stream.mp4"
        run_cmd(
            [
                "ffmpeg", "-y", "-v", "error",
                "-f", "lavfi", "-i", "testsrc=duration=6:size=1280x720:rate=30",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                str(src_vid),
            ]
        )
        pcm, db = extract_16k_wav(src_vid, tmp / "audio.wav")
        mock_words = [
            {"w": "Bro", "s": 0.2, "e": 0.6},
            {"w": "look", "s": 0.65, "e": 1.0},
            {"w": "at", "s": 1.05, "e": 1.3},
            {"w": "Vice", "s": 1.35, "e": 1.8},
            {"w": "City", "s": 1.85, "e": 2.4},
            {"w": "police", "s": 2.5, "e": 3.1},
            {"w": "chase!", "s": 3.15, "e": 3.9},
        ]
        clip_spec = {
            "start": 0.0,
            "end": 4.5,
            "score": 92,
            "hook": "6 STAR COP CHASE",
            "highlight_words": ["Vice", "City", "police"],
        }
        rendered = render_vertical_short(
            src_mp4=src_vid,
            out_dir=tmp / "clips",
            stem="test_short_01",
            clip=clip_spec,
            words=mock_words,
            src_w=1280,
            src_h=720,
            layout="blur",
            style_name="bounce",
            isolate_voice_band=True,
        )
        info = probe_video(rendered["mp4"])
        print(f"Rendered 9:16 Short: {rendered['mp4'].name} ({info['width']}x{info['height']}, {info['duration']:.2f}s)")
        print(f"Rendered 3:2-Safe Thumbnail: {rendered['thumb'].name} ({rendered['thumb'].stat().st_size} bytes)")
        assert info["width"] == 1080 and info["height"] == 1920, "Output must be 1080x1920 vertical"
        print("UPGRADED STEP 2 SELF-TEST PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Step 2 (Upgraded): Stream/Drive Clipper, 9:16 ASS Renderer & Drive Output Sync")
    ap.add_argument("--url", type=str, default="", help="Twitch, Kick, or YouTube Live/VOD URL to clip")
    ap.add_argument("--fallback-url", type=str, default="", help="Fallback Twitch/Kick simulcast URL if YouTube blocks cloud IP")
    ap.add_argument("--auto-24x7", action="store_true", help="Full 24/7 autonomous mode: Drive Inbox -> Preferred YouTube Channels -> Live Search -> Analytics Heatmap Schedule")
    ap.add_argument("--topic", type=str, default="", help="Optional topic/niche override (e.g. 'GTA 6', 'Podcast', 'Cricket', or 'auto')")
    ap.add_argument("--from-drive", action="store_true", help="Pull next unclipped video from GDRIVE_INBOX_FOLDER_ID")
    ap.add_argument("--save-to-drive", action="store_true", help="Upload finished Shorts & Thumbnails to GDRIVE_OUTPUT_FOLDER_ID")
    ap.add_argument("--slice-minutes", type=float, default=15.0, help="Minutes of live stream to capture")
    ap.add_argument("--max-clips", type=int, default=2, help="Max vertical Shorts to render per run")
    ap.add_argument("--layout", choices=["blur", "zoom", "crop"], default="blur", help="9:16 framing layout")
    ap.add_argument("--style", choices=["bounce", "hype", "clean"], default="bounce", help="ASS caption animation style")
    ap.add_argument("--isolate-voice", action="store_true", help="Filter background radio music frequencies")
    ap.add_argument("--schedule", action="store_true", help="Auto-upload and schedule on YouTube via Step 1")
    ap.add_argument("--self-test", action="store_true", help="Run end-to-end FFmpeg + ASS + Thumbnail render verification")
    args = ap.parse_args()

    if args.self_test:
        run_self_test()
        raise SystemExit(0)

    work = Path("work/run_" + str(int(time.time())))
    work.mkdir(parents=True, exist_ok=True)

    # 0. Load persistent 24/7 cloud state ledger & update YouTube Analytics Heatmap
    cloud_state = load_cloud_state()
    daily_plan = research_daily_viral_trends(cloud_state, topic_hint=args.topic or env('TOPIC_NICHE', 'auto'))
    if args.schedule or args.auto_24x7:
        try:
            cloud_state = update_analytics_heatmap(state=cloud_state)
        except Exception as exc:
            print(f"[WARN] Analytics heatmap refresh skipped: {exc}")

    # 1. Ingest: Drive Inbox -> Manual URL -> 24/7 Auto-Discovery (Preferred YouTube Channels + Live Search)
    ingested = None
    dedup_key = ""
    if args.from_drive or (not args.url and env("GDRIVE_INBOX_FOLDER_ID")):
        try:
            ingested = ingest_from_google_drive(work)
        except Exception as exc:
            print(f"[WARN] Drive Inbox check skipped: {exc}")

    target_url = args.url.strip()
    fallback_url = args.fallback_url.strip()
    discovered_meta = None
    if ingested is None and not target_url and args.auto_24x7:
        discovered_meta = discover_active_stream(state=cloud_state)
        if discovered_meta:
            target_url = discovered_meta["url"]
            fallback_url = discovered_meta.get("fallback_url", "")
            dedup_key = discovered_meta.get("dedup_key", "")

    if ingested is None and target_url:
        try:
            ingested = capture_stream_slice(target_url, work, slice_minutes=args.slice_minutes)
        except Exception as exc:
            if fallback_url:
                print(f"[FALLBACK] Primary URL refused ({exc}); switching to simulcast {fallback_url}")
                ingested = capture_stream_slice(fallback_url, work, slice_minutes=args.slice_minutes)
                if discovered_meta and discovered_meta.get("uploader"):
                    ingested["uploader"] = discovered_meta["uploader"]
                    ingested["title"] = discovered_meta["title"]
            else:
                raise

    if ingested is None:
        print("[IDLE] No unclipped Drive video or active stream found this cycle. Sleeping until next cron run.")
        save_cloud_state(cloud_state)
        raise SystemExit(0)

    # 2. Extract audio, transcribe, and pick top viral moments
    pcm, db = extract_16k_wav(ingested["video_path"], work / "audio.wav")
    words = ingested["native_words"] or transcribe_audio(work / "audio.wav", pcm)
    picks = pick_viral_gta6_moments(words, db, ingested["title"], ingested["uploader"], max_clips=args.max_clips, chat_buckets=ingested.get("chat_buckets"))

    should_save_drive = args.save_to_drive or bool(env("GDRIVE_OUTPUT_FOLDER_ID"))
    scheduled_slots: List[str] = list(cloud_state.get("scheduled_iso_slots") or [])
    golden_hours: List[int] = list(cloud_state.get("golden_hours_utc") or [12, 15, 18, 21, 0, 9])

    for idx, clip in enumerate(picks, start=1):
        ts_label = int(time.time())
        stem = f"gta6_{re.sub(r'[^a-zA-Z0-9]', '', ingested['uploader'])[:12]}_{ts_label}_{idx:02d}"
        out = render_vertical_short(
            src_mp4=ingested["video_path"],
            out_dir=Path("work/clips"),
            stem=stem,
            clip=clip,
            words=words,
            src_w=ingested["width"],
            src_h=ingested["height"],
            layout=args.layout,
            style_name=args.style,
            isolate_voice_band=args.isolate_voice,
        )
        clip_transcript = " ".join(w["w"] for w in words if clip["start"] <= w["s"] <= clip["end"])
        seo = generate_gta6_seo_metadata(
            streamer=ingested["uploader"],
            stream_title=ingested["title"],
            hook=clip.get("hook", ""),
            transcript=clip_transcript,
        )
        build_shorts_thumbnail(out["thumb"], out["thumb"], seo.get("thumbnail_badge") or clip.get("hook", ""))
        slot_iso = compute_next_publish_slot(
            gap_hours=float(env("POST_GAP_HOURS", "3")),
            existing_iso_slots=scheduled_slots,
            golden_hours_utc=golden_hours,
        )
        scheduled_slots.append(slot_iso)

        # Save SEO metadata JSON sidecar
        seo_path = Path("work/clips") / f"{stem}_seo.json"
        seo_payload = {**seo, "publishAt": slot_iso, "streamer": ingested["uploader"], "source_title": ingested["title"]}
        seo_path.write_text(json.dumps(seo_payload, indent=2), encoding="utf-8")

        print(f"[{stem}] Rendered: {out['mp4'].name} | Title: {seo['title']} | Slot: {slot_iso}")

        # 3. Save finished Short, Thumbnail & SEO JSON back to Google Drive Output folder
        if should_save_drive and env("GDRIVE_OUTPUT_FOLDER_ID"):
            vid_drive_id, vid_drive_url = drive_upload_file(
                local_path=out["mp4"],
                mime_type="video/mp4",
                description=f"{seo['title']}\nScheduled: {slot_iso}\n\n{seo['description']}",
            )
            thumb_drive_id, thumb_drive_url = drive_upload_file(
                local_path=out["thumb"],
                mime_type="image/jpeg",
                description=f"Thumbnail for {seo['title']}",
            )
            drive_upload_file(
                local_path=seo_path,
                mime_type="application/json",
                description=f"SEO metadata for {seo['title']}",
            )
            print(f"[{stem}] Saved to Google Drive: {vid_drive_url} | Thumbnail: {thumb_drive_url}")

        # 4. Upload and schedule directly on YouTube Studio
        if args.schedule:
            vid_id, url_short = upload_and_schedule_short(
                video_path=out["mp4"],
                thumb_path=out["thumb"],
                title=seo["title"],
                description=seo["description"],
                tags=seo["tags"],
                publish_at_iso=slot_iso,
            )
            print(f"[{stem}] Scheduled on YouTube Studio: {url_short} (publishAt={slot_iso})")
            if seo.get("pinned_comment"):
                post_engagement_comment(vid_id, seo["pinned_comment"])
            send_telegram_summary_card(seo["title"], ingested["uploader"], slot_iso, yt_url=url_short)

            # 4b. Auto-schedule on Facebook Page Reels (if FB_PAGE_ID & FB_PAGE_ACCESS_TOKEN are set)
            social_cap = build_social_caption(seo, ingested["uploader"])
            if env("FB_PAGE_ID") and env("FB_PAGE_ACCESS_TOKEN"):
                try:
                    fb_id, fb_url = upload_facebook_reel(
                        video_path=out["mp4"],
                        title=seo["title"],
                        caption=social_cap,
                        publish_at_iso=slot_iso,
                    )
                    print(f"[{stem}] Scheduled on Facebook Reels: {fb_url} (slot={slot_iso})")
                except Exception as exc:
                    print(f"[WARN] Facebook Reels upload skipped/failed: {exc}")

            # 4c. Auto-publish on Instagram Reels (if IG_USER_ID is set)
            if env("IG_USER_ID") and (env("IG_ACCESS_TOKEN") or env("FB_PAGE_ACCESS_TOKEN")):
                try:
                    ig_id, ig_url = upload_instagram_reel(
                        video_path=out["mp4"],
                        caption=social_cap,
                    )
                    print(f"[{stem}] Published on Instagram Reels: {ig_url}")
                except Exception as exc:
                    print(f"[WARN] Instagram Reels upload skipped/failed: {exc}")

    # 5. If source came from Google Drive Inbox, mark it `[CLIPPED]` so it isn't repeated
    if ingested.get("drive_file_id"):
        drive_mark_processed(ingested["drive_file_id"], ingested["drive_file_name"])
        print(f"[DRIVE INBOX] Marked '{ingested['drive_file_name']}' as [CLIPPED].")

    # 6. Save updated deduplication ledger, heatmap & scheduled slots to Google Drive
    if dedup_key:
        cloud_state.setdefault("processed_keys", []).append(dedup_key)
    cloud_state["scheduled_iso_slots"] = scheduled_slots[-60:]
    save_cloud_state(cloud_state)
    print(f"[STATE SYNCED] Golden Hours (UTC): {golden_hours} | Total tracked slots: {len(scheduled_slots)}")
