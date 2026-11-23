"""
YouTube Vertical Live (9:16) Autonomous Highlight Streamer
"""
import argparse
import datetime as dt
import os
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
from PIL import Image, ImageDraw, ImageFont

from youtube_seo_scheduler import (
    DRIVE_FILES_URL,
    _get_or_create_app_drive_folder,
    call_gemini_json_with_cascade,
    drive_download_file,
    env,
    extract_drive_folder_id,
    refresh_google_access_token,
)

YT_LIVE_BROADCASTS_URL = "https://www.googleapis.com/youtube/v3/liveBroadcasts"
YT_LIVE_STREAMS_URL = "https://www.googleapis.com/youtube/v3/liveStreams"
YT_THUMB_URL = "https://www.googleapis.com/upload/youtube/v3/thumbnails/set"
YT_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
OUT_W, OUT_H, FPS = 1080, 1920, 30


def run_cmd(cmd: List[str], timeout: int = 900) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"{Path(str(cmd[0])).name} failed: {p.stderr[-900:]}")
    return p.stdout


def _load_bold_font(size: int) -> ImageFont.FreeTypeFont:
    for cand in [
        Path("fonts/Poppins-ExtraBold.ttf"),
        Path("fonts/Anton-Regular.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"),
    ]:
        if cand.is_file():
            try:
                return ImageFont.truetype(str(cand), size)
            except OSError:
                pass
    return ImageFont.load_default()


def build_live_overlay_png(out_png: Path, hook_text: str = "LIVE GTA 6 TOP CLIPS • SUBSCRIBE TO HIT 1K!") -> Path:
    im = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im, "RGBA")
    draw.rectangle([0, 0, OUT_W, 92], fill=(9, 10, 16, 220))
    draw.rectangle([0, 88, OUT_W, 94], fill=(255, 42, 133, 255))
    draw.rounded_rectangle([28, 20, 168, 72], radius=14, fill=(235, 30, 50, 255))
    live_font = _load_bold_font(34)
    lb = draw.textbbox((0, 0), "LIVE", font=live_font)
    draw.text((98 - (lb[2] - lb[0]) // 2, 46 - (lb[3] - lb[1]) // 2 - lb[1]), "LIVE", font=live_font, fill=(255, 255, 255, 255))

    txt = hook_text.upper()[:46]
    fsize = 34
    while fsize >= 22:
        tfont = _load_bold_font(fsize)
        tb = draw.textbbox((0, 0), txt, font=tfont)
        if (tb[2] - tb[0]) <= (OUT_W - 210):
            break
        fsize -= 2
    tb = draw.textbbox((0, 0), txt, font=tfont)
    draw.text((192, 46 - (tb[3] - tb[1]) // 2 - tb[1]), txt, font=tfont, fill=(255, 183, 3, 255))
    im.save(out_png, format="PNG")
    return out_png


def build_live_thumbnail(sample_mp4: Path, out_jpg: Path, work: Path, badge: str = "LIVE GTA 6 CLIPS") -> Path:
    raw_jpg = work / "live_raw_frame.jpg"
    run_cmd(["ffmpeg", "-y", "-v", "error", "-ss", "1.5", "-i", str(sample_mp4), "-frames:v", "1", "-q:v", "3", str(raw_jpg)], timeout=30)
    im = Image.open(raw_jpg).convert("RGB").resize((OUT_W, OUT_H), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(im, "RGBA")
    text = badge.upper()[:24]
    font = _load_bold_font(76)
    bb = draw.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    bx0, by0 = (OUT_W - tw) // 2 - 48, 960 - th // 2 - 28
    bx1, by1 = bx0 + tw + 96, by0 + th + 56
    draw.rounded_rectangle([bx0 - 6, by0 + 8, bx1 + 6, by1 + 12], radius=26, fill=(0, 0, 0, 190))
    draw.rounded_rectangle([bx0, by0, bx1, by1], radius=24, fill=(255, 42, 133, 248), outline=(255, 183, 3, 255), width=6)
    draw.text(((OUT_W - tw) // 2, by0 + 24 - bb[1]), text, font=font, fill=(255, 255, 255, 255))
    im.save(out_jpg, format="JPEG", quality=92)
    return out_jpg


def normalize_clip_for_rtmp(src_mp4: Path, overlay_png: Path, out_mp4: Path) -> Path:
    run_cmd([
        "ffmpeg", "-y", "-v", "error", "-i", str(src_mp4), "-i", str(overlay_png),
        "-filter_complex", f"[0:v]fps={FPS},scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,crop={OUT_W}:{OUT_H},setsar=1[bg];[bg][1:v]overlay=0:0[v]",
        "-map", "[v]", "-map", "0:a:0",
        "-c:v", "libx264", "-preset", "veryfast", "-b:v", "4200k", "-maxrate", "4500k", "-bufsize", "8400k",
        "-g", "60", "-keyint_min", "60", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2", str(out_mp4),
    ])
    return out_mp4


def list_drive_shorts_for_live(max_clips: int = 15) -> List[Dict]:
    token = refresh_google_access_token()
    folder_ids: List[str] = []
    env_fid = extract_drive_folder_id(env("GDRIVE_OUTPUT_FOLDER_ID"))
    if env_fid:
        folder_ids.append(env_fid)
    app_fid = _get_or_create_app_drive_folder("RazonGamerz_Output", token)
    if app_fid and app_fid not in folder_ids:
        folder_ids.append(app_fid)
    found, seen = [], set()
    for fid in folder_ids:
        q = f"'{fid}' in parents and trashed = false and mimeType = 'video/mp4' and not name contains 'weekly_compilation'"
        r = httpx.get(DRIVE_FILES_URL, params={"q": q, "orderBy": "createdTime desc", "pageSize": max_clips, "fields": "files(id, name, description)"}, headers={"Authorization": f"Bearer {token}"}, timeout=30)
        if r.status_code == 200:
            for item in r.json().get("files") or []:
                if item["id"] not in seen:
                    seen.add(item["id"])
                    found.append(item)
    return found[:max_clips]


def generate_live_seo(clip_titles: List[str], rotation_idx: int = 0) -> Dict[str, str]:
    fallback_titles = [
        "🔴 LIVE: GTA 6 Craziest Moments & Reactions! (Sub Goal: 1,000) #GTA6 #Shorts",
        "🔴 LIVE: IShowSpeed, Kai Cenat & Top GTA 6 Clips Non-Stop! #GTA6 #GamingLive",
        "🔴 LIVE: Wildest GTA 6 Vice City Fails & Clutches! Drop a Sub! #GTA6 #Live",
    ]
    fb_title = fallback_titles[rotation_idx % len(fallback_titles)]
    fb_desc = "Welcome to RazonGamerz Vertical Live! Non-stop edited GTA 6 & top gaming stream highlights.\n\nHit SUBSCRIBE to help us reach 1,000 subscribers!\n#GTA6 #GamingLive #RazonGamerz #Shorts"
    try:
        prompt = (
            f"Write a high-CTR YouTube Vertical Live stream title (max 90 chars, starts with '🔴 LIVE:', includes a subscriber/curiosity hook and #GTA6) "
            f"and a 3-sentence live description for RazonGamerz. Rotation #{rotation_idx + 1}. Featured clips:\n" + "\n".join(clip_titles[:8])
        )
        res = call_gemini_json_with_cascade(
            prompt,
            {"type": "OBJECT", "properties": {"title": {"type": "STRING"}, "description": {"type": "STRING"}, "badge": {"type": "STRING"}}, "required": ["title", "description", "badge"]},
        )
        return {
            "title": str(res.get("title") or fb_title).strip()[:95],
            "description": str(res.get("description") or fb_desc).strip()[:4500],
            "badge": str(res.get("badge") or "LIVE GTA 6 CLIPS").strip().upper()[:24],
        }
    except Exception:
        return {"title": fb_title, "description": fb_desc, "badge": "LIVE GTA 6 CLIPS"}


def create_youtube_vertical_broadcast(title: str, description: str, thumb_path: Path) -> Tuple[str, str]:
    token = refresh_google_access_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    start_iso = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=15)).strftime("%Y-%m-%dT%H:%M:%SZ")

    b_res = httpx.post(
        YT_LIVE_BROADCASTS_URL, params={"part": "snippet,status,contentDetails"}, headers=headers,
        json={
            "snippet": {"title": title, "description": description, "scheduledStartTime": start_iso},
            "status": {"privacyStatus": "public", "selfDeclaredMadeForKids": False},
            "contentDetails": {"enableAutoStart": True, "enableAutoStop": True, "latencyPreference": "normal"},
        }, timeout=30,
    )
    if b_res.status_code >= 400:
        raise RuntimeError(f"liveBroadcasts.insert failed ({b_res.status_code}): {b_res.text[:400]}")
    broadcast_id = b_res.json()["id"]

    s_res = httpx.post(
        YT_LIVE_STREAMS_URL, params={"part": "snippet,cdn"}, headers=headers,
        json={"snippet": {"title": f"RazonGamerz Vertical Stream {start_iso}"}, "cdn": {"frameRate": "variable", "ingestionType": "rtmp", "resolution": "variable"}}, timeout=30,
    )
    if s_res.status_code >= 400:
        raise RuntimeError(f"liveStreams.insert failed ({s_res.status_code}): {s_res.text[:400]}")
    stream_data = s_res.json()
    stream_id = stream_data["id"]
    ing = stream_data["cdn"]["ingestionInfo"]
    base_url = ing.get("rtmpsIngestionAddress") or ing["ingestionAddress"]
    rtmp_full = f"{base_url.rstrip('/')}/{ing['streamName']}"

    bind_res = httpx.post(f"{YT_LIVE_BROADCASTS_URL}/bind", params={"id": broadcast_id, "part": "id,contentDetails", "streamId": stream_id}, headers=headers, timeout=30)
    if bind_res.status_code >= 400:
        raise RuntimeError(f"liveBroadcasts.bind failed ({bind_res.status_code}): {bind_res.text[:300]}")

    if thumb_path.is_file():
        httpx.post(YT_THUMB_URL, params={"videoId": broadcast_id}, headers={"Authorization": f"Bearer {token}", "Content-Type": "image/jpeg"}, content=thumb_path.read_bytes(), timeout=60)

    print(f"[YOUTUBE VERTICAL LIVE] Broadcast ready: https://www.youtube.com/watch?v={broadcast_id}")
    return broadcast_id, rtmp_full


def update_live_metadata(broadcast_id: str, title: str, description: str) -> None:
    try:
        token = refresh_google_access_token()
        r = httpx.put(
            YT_VIDEOS_URL, params={"part": "snippet"},
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"id": broadcast_id, "snippet": {"title": title[:95], "description": description[:4500], "categoryId": "20"}}, timeout=25,
        )
        if r.status_code < 400:
            print(f"[GEMINI LIVE OPTIMIZER] Rotated live title -> {title}")
    except Exception as exc:
        print(f"[WARN] Live title rotation skipped: {exc}")


def end_youtube_broadcast(broadcast_id: str) -> None:
    try:
        token = refresh_google_access_token()
        httpx.post(f"{YT_LIVE_BROADCASTS_URL}/transition", params={"broadcastStatus": "complete", "id": broadcast_id, "part": "id,status"}, headers={"Authorization": f"Bearer {token}"}, timeout=25)
        print(f"[YOUTUBE VERTICAL LIVE] Cleanly ended broadcast {broadcast_id}")
    except Exception:
        pass


def run_self_test() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="vlive_test_"))
    try:
        sample = tmp / "sample.mp4"
        run_cmd(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=duration=3:size=1080x1920:rate={FPS}", "-f", "lavfi", "-i", "sine=frequency=440:duration=3:sample_rate=44100", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", str(sample)])
        ov = build_live_overlay_png(tmp / "overlay.png")
        norm = normalize_clip_for_rtmp(sample, ov, tmp / "norm.mp4")
        th = build_live_thumbnail(norm, tmp / "live_thumb.jpg", tmp)
        assert norm.stat().st_size > 10000 and th.stat().st_size > 10000
        print("YOUTUBE VERTICAL LIVE SELF-TEST PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="YouTube Vertical Live (9:16) Highlight Streamer")
    ap.add_argument("--duration-minutes", type=int, default=180)
    ap.add_argument("--max-clips", type=int, default=15)
    ap.add_argument("--rotate-minutes", type=int, default=60)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        run_self_test()
        raise SystemExit(0)

    items = list_drive_shorts_for_live(max_clips=args.max_clips)
    if not items:
        print("[VERTICAL LIVE] No edited Shorts found in RazonGamerz_Output Drive folder. Exiting.")
        raise SystemExit(0)

    work = Path(tempfile.mkdtemp(prefix="vlive_run_"))
    broadcast_id = ""
    try:
        overlay_png = build_live_overlay_png(work / "live_overlay.png")
        norm_clips, clip_titles = [], []
        for idx, item in enumerate(items, start=1):
            raw, norm = work / f"raw_{idx:02d}.mp4", work / f"norm_{idx:02d}.mp4"
            drive_download_file(item["id"], raw)
            normalize_clip_for_rtmp(raw, overlay_png, norm)
            raw.unlink(missing_ok=True)
            norm_clips.append(norm)
            first_line = (item.get("description") or "").splitlines()[0].strip() if item.get("description") else item["name"]
            clip_titles.append(re.sub(r"#\w+", "", first_line).strip())

        seo0 = generate_live_seo(clip_titles, rotation_idx=0)
        live_thumb = build_live_thumbnail(norm_clips[0], work / "live_thumb.jpg", work, badge=seo0["badge"])
        broadcast_id, rtmp_url = create_youtube_vertical_broadcast(seo0["title"], seo0["description"], live_thumb)

        playlist_txt = work / "playlist.txt"
        playlist_txt.write_text("\n".join(f"file '{p.resolve()}'" for p in norm_clips), encoding="utf-8")

        dur_sec = max(60, min(300, args.duration_minutes) * 60)
        stop_ev = threading.Event()

        def _title_rotator():
            rot = 1
            while not stop_ev.wait(max(300, args.rotate_minutes * 60)):
                seo_n = generate_live_seo(clip_titles, rotation_idx=rot)
                update_live_metadata(broadcast_id, seo_n["title"], seo_n["description"])
                rot += 1

        threading.Thread(target=_title_rotator, daemon=True).start()
        print(f"[VERTICAL LIVE] Streaming {len(norm_clips)} clips in 9:16 loop for {args.duration_minutes} min...")
        proc = subprocess.Popen([
            "ffmpeg", "-y", "-v", "error", "-re", "-stream_loop", "-1",
            "-f", "concat", "-safe", "0", "-i", str(playlist_txt),
            "-t", str(dur_sec), "-c:v", "copy", "-c:a", "copy", "-f", "flv", rtmp_url,
        ])
        try:
            proc.wait(timeout=dur_sec + 120)
        finally:
            stop_ev.set()
            if proc.poll() is None:
                proc.terminate()
            end_youtube_broadcast(broadcast_id)
    finally:
        shutil.rmtree(work, ignore_errors=True)
