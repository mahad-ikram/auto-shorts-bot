"""
Upgrade #4 — Sunday Auto-Compiled 16:9 YouTube Highlight Compilation Builder
"""
import argparse
import datetime as dt
import os
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from youtube_seo_scheduler import (
    DRIVE_FILES_URL,
    _get_or_create_app_drive_folder,
    call_gemini_json_with_cascade,
    drive_download_file,
    drive_upload_file,
    env,
    extract_drive_folder_id,
    load_cloud_state,
    save_cloud_state,
    refresh_google_access_token,
    upload_and_schedule_short,
)

W_OUT, H_OUT, FPS = 1920, 1080, 30


def run_cmd(cmd: List[str], timeout: int = 1200) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"{Path(str(cmd[0])).name} failed: {p.stderr[-900:]}")
    return p.stdout


def probe_duration(path: Path) -> float:
    out = run_cmd(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)])
    return float(out.strip() or 0.0)


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


def _make_bumper_sfx(out_wav: Path, sr: int = 48000) -> Path:
    dur = 1.5
    t = np.linspace(0.0, dur, int(sr * dur), endpoint=False, dtype=np.float32)
    freq = 140.0 + 420.0 * (1.0 - np.exp(-10.0 * t))
    sig = np.sin(2.0 * np.pi * np.cumsum(freq) / sr) * np.exp(-9.0 * t) * 0.35
    pcm = (np.clip(sig, -0.95, 0.95) * 32767.0).astype(np.int16)
    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(np.column_stack((pcm, pcm)).ravel().tobytes())
    return out_wav


def render_numbered_bumper(index_num: int, clip_title: str, out_mp4: Path, work: Path) -> Path:
    img = Image.new("RGB", (W_OUT, H_OUT), (9, 10, 16))
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle([0, 0, W_OUT, 18], fill=(255, 42, 133, 255))
    draw.rectangle([0, H_OUT - 18, W_OUT, H_OUT], fill=(123, 44, 191, 255))

    badge_font, title_font, sub_font = _load_bold_font(120), _load_bold_font(64), _load_bold_font(36)
    num_txt = f"#{index_num:02d}"
    nb = draw.textbbox((0, 0), num_txt, font=badge_font)
    nw, nh = nb[2] - nb[0], nb[3] - nb[1]
    bx0, by0 = (W_OUT - nw) // 2 - 60, 270
    bx1, by1 = bx0 + nw + 120, by0 + nh + 56
    draw.rounded_rectangle([bx0, by0, bx1, by1], radius=28, fill=(255, 42, 133, 250), outline=(255, 183, 3, 255), width=6)
    draw.text(((W_OUT - nw) // 2, by0 + 24 - nb[1]), num_txt, font=badge_font, fill=(255, 255, 255, 255))

    import textwrap
    clean_t = re.sub(r"#\w+", "", clip_title or f"Highlight #{index_num}").strip()[:64].upper()
    lines = textwrap.wrap(clean_t, width=28)[:2] or [clean_t]
    fsize = 64
    while fsize >= 38:
        title_font = _load_bold_font(fsize)
        if all((draw.textbbox((0, 0), ln, font=title_font)[2] - draw.textbbox((0, 0), ln, font=title_font)[0]) <= (W_OUT - 220) for ln in lines):
            break
        fsize -= 4
    y_cursor = by1 + 65
    for ln in lines:
        tb = draw.textbbox((0, 0), ln, font=title_font)
        draw.text(((W_OUT - (tb[2] - tb[0])) // 2, y_cursor), ln, font=title_font, fill=(248, 249, 250, 255))
        y_cursor += (tb[3] - tb[1]) + 26

    brand_txt = "RAZONGAMERZ  •  WEEKLY TOP CLIPS"
    bb = draw.textbbox((0, 0), brand_txt, font=sub_font)
    draw.text(((W_OUT - (bb[2] - bb[0])) // 2, H_OUT - 120), brand_txt, font=sub_font, fill=(255, 183, 3, 255))

    card_png = work / f"card_{index_num:02d}.png"
    img.save(card_png)
    sfx_wav = _make_bumper_sfx(work / f"card_{index_num:02d}.wav")

    run_cmd([
        "ffmpeg", "-y", "-v", "error", "-loop", "1", "-t", "1.5", "-i", str(card_png), "-i", str(sfx_wav),
        "-c:v", "libx264", "-r", str(FPS), "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "21",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2", "-shortest", str(out_mp4),
    ])
    return out_mp4


def convert_clip_to_16x9(src_mp4: Path, out_mp4: Path) -> Path:
    vf = (
        f"[0:v]fps={FPS},split=2[bg_in][fg_in];"
        f"[bg_in]scale=480:270:force_original_aspect_ratio=increase,crop=480:270,boxblur=10:3,"
        f"scale={W_OUT}:{H_OUT},eq=brightness=-0.18:saturation=1.25[bg];"
        f"[fg_in]scale=-2:{H_OUT}:force_original_aspect_ratio=decrease[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[v]"
    )
    run_cmd([
        "ffmpeg", "-y", "-v", "error", "-i", str(src_mp4), "-filter_complex", vf,
        "-map", "[v]", "-map", "0:a:0",
        "-af", "aformat=sample_rates=48000:channel_layouts=stereo,loudnorm=I=-14:TP=-1.5:LRA=11",
        "-c:v", "libx264", "-r", str(FPS), "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "21",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2", str(out_mp4),
    ])
    return out_mp4


def build_compilation_thumbnail(clip_paths: List[Path], out_jpg: Path, work: Path, headline: str = "TOP MOMENTS OF THE WEEK") -> Path:
    canvas = Image.new("RGB", (1280, 720), (9, 10, 16))
    panels = clip_paths[:3]
    panel_w = 1280 // max(1, len(panels))
    for i, cp in enumerate(panels):
        f_jpg = work / f"thumb_src_{i}.jpg"
        try:
            run_cmd(["ffmpeg", "-y", "-v", "error", "-ss", "2.0", "-i", str(cp), "-frames:v", "1", "-q:v", "3", str(f_jpg)], timeout=30)
            im = Image.open(f_jpg).convert("RGB")
            ratio = max(panel_w / im.width, 720 / im.height)
            im = im.resize((int(im.width * ratio) + 2, int(im.height * ratio) + 2), Image.Resampling.LANCZOS)
            left, top = (im.width - panel_w) // 2, (im.height - 720) // 2
            canvas.paste(im.crop((left, top, left + panel_w, top + 720)), (i * panel_w, 0))
        except Exception:
            continue

    draw = ImageDraw.Draw(canvas, "RGBA")
    for x in range(1, len(panels)):
        draw.rectangle([x * panel_w - 4, 0, x * panel_w + 4, 720], fill=(255, 183, 3, 255))

    text = headline.upper()[:30]
    font = _load_bold_font(68)
    bb = draw.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    bx0, by0 = (1280 - tw) // 2 - 44, 540
    bx1, by1 = bx0 + tw + 88, by0 + th + 40
    draw.rounded_rectangle([bx0 - 6, by0 + 8, bx1 + 6, by1 + 12], radius=24, fill=(0, 0, 0, 195))
    draw.rounded_rectangle([bx0, by0, bx1, by1], radius=22, fill=(255, 42, 133, 248), outline=(255, 183, 3, 255), width=6)
    draw.text(((1280 - tw) // 2, by0 + 18 - bb[1]), text, font=font, fill=(255, 255, 255, 255))
    canvas.save(out_jpg, format="JPEG", quality=92)
    return out_jpg


def _fmt_chapter_ts(sec: float) -> str:
    s = max(0, int(round(sec)))
    return f"{s // 60:02d}:{s % 60:02d}"


def list_recent_output_clips(days: int = 7, max_clips: int = 15) -> List[Dict]:
    token = refresh_google_access_token()
    folder_ids: List[str] = []
    env_fid = extract_drive_folder_id(env("GDRIVE_OUTPUT_FOLDER_ID"))
    if env_fid:
        folder_ids.append(env_fid)
    app_fid = _get_or_create_app_drive_folder("RazonGamerz_Output", token)
    if app_fid and app_fid not in folder_ids:
        folder_ids.append(app_fid)

    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    found, seen_ids = [], set()
    for fid in folder_ids:
        q = f"'{fid}' in parents and trashed = false and mimeType = 'video/mp4' and createdTime >= '{cutoff}' and not name contains 'weekly_compilation'"
        r = httpx.get(DRIVE_FILES_URL, params={"q": q, "orderBy": "createdTime desc", "pageSize": max_clips, "fields": "files(id, name, description, createdTime)"}, headers={"Authorization": f"Bearer {token}"}, timeout=30)
        if r.status_code == 200:
            for item in r.json().get("files") or []:
                if item["id"] not in seen_ids:
                    seen_ids.add(item["id"])
                    found.append(item)
    return found[:max_clips]


def run_self_test() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="weekly_test_"))
    try:
        clips = []
        for i in range(1, 4):
            p = tmp / f"clip_{i}.mp4"
            run_cmd(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=duration=3:size=1080x1920:rate={FPS}", "-f", "lavfi", "-i", f"sine=frequency={330 + i * 110}:duration=3:sample_rate=48000", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", str(p)])
            clips.append((p, f"Test Viral Moment #{i}"))
        segments, chapters, cursor = [], [], 0.0
        for idx, (cp, title) in enumerate(clips, 1):
            chapters.append(f"{_fmt_chapter_ts(cursor)} #{idx} - {title}")
            bump = render_numbered_bumper(idx, title, tmp / f"bump_{idx}.mp4", tmp)
            wide = convert_clip_to_16x9(cp, tmp / f"wide_{idx}.mp4")
            segments.extend([bump, wide])
            cursor += probe_duration(bump) + probe_duration(wide)
        concat_txt = tmp / "concat.txt"
        concat_txt.write_text("\n".join(f"file '{s.resolve()}'" for s in segments), encoding="utf-8")
        final_mp4 = tmp / "weekly_compilation.mp4"
        run_cmd(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(concat_txt), "-c", "copy", "-movflags", "+faststart", str(final_mp4)])
        thumb = build_compilation_thumbnail([c[0] for c in clips], tmp / "weekly_thumb.jpg", tmp)
        assert probe_duration(final_mp4) >= 12.0 and thumb.stat().st_size > 10000
        print("UPGRADE #4 WEEKLY COMPILER SELF-TEST PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Upgrade #4: Sunday Auto-Compiled 16:9 YouTube Highlight Video")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--max-clips", type=int, default=15)
    ap.add_argument("--min-clips", type=int, default=3)
    ap.add_argument("--schedule", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        run_self_test()
        raise SystemExit(0)

    items = list_recent_output_clips(days=args.days, max_clips=args.max_clips)
    if len(items) < args.min_clips:
        print(f"[WEEKLY COMPILER] Only {len(items)} clips found in last {args.days} days (need {args.min_clips}). Skipping.")
        raise SystemExit(0)

    work = Path(tempfile.mkdtemp(prefix="weekly_run_"))
    try:
        segments, raw_clips, chapters, cursor = [], [], [], 0.0
        for idx, item in enumerate(items, start=1):
            local_raw = work / f"raw_{idx:02d}.mp4"
            drive_download_file(item["id"], local_raw)
            raw_clips.append(local_raw)
            first_line = (item.get("description") or "").splitlines()[0].strip() if item.get("description") else item["name"]
            clean_title = re.sub(r"#\w+", "", first_line).strip() or f"Highlight #{idx}"
            chapters.append(f"{_fmt_chapter_ts(cursor)} #{idx} - {clean_title[:60]}")
            bump = render_numbered_bumper(idx, clean_title, work / f"bump_{idx:02d}.mp4", work)
            wide = convert_clip_to_16x9(local_raw, work / f"wide_{idx:02d}.mp4")
            segments.extend([bump, wide])
            cursor += probe_duration(bump) + probe_duration(wide)

        concat_txt = work / "concat.txt"
        concat_txt.write_text("\n".join(f"file '{s.resolve()}'" for s in segments), encoding="utf-8")
        week_tag = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d")
        out_dir = Path("work/compilations")
        out_dir.mkdir(parents=True, exist_ok=True)
        final_mp4 = out_dir / f"weekly_compilation_{week_tag}.mp4"
        final_thumb = out_dir / f"weekly_compilation_{week_tag}_thumb.jpg"

        run_cmd(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(concat_txt), "-c", "copy", "-movflags", "+faststart", str(final_mp4)])
        build_compilation_thumbnail(raw_clips, final_thumb, work, headline=f"TOP {len(items)} CLIPS OF THE WEEK")

        chapter_block = "\n".join(chapters)
        yt_title = f"Top {len(items)} Craziest GTA 6 & Gaming Stream Moments of the Week! | RazonGamerz"[:95]
        yt_desc = f"The {len(items)} wildest live gaming reactions and GTA 6 moments from this week—compiled in full HD by RazonGamerz!\n\nTimestamps / Chapters:\n{chapter_block}\n\nSubscribe to @RazonGamerz!\n#GTA6 #GamingHighlights #RazonGamerz"
        try:
            seo = call_gemini_json_with_cascade(
                f"Write a high-CTR YouTube long-form compilation title (max 85 chars) and 2-sentence intro hook for these {len(items)} weekly gaming highlights:\n{chapter_block}",
                {"type": "OBJECT", "properties": {"title": {"type": "STRING"}, "intro": {"type": "STRING"}}, "required": ["title", "intro"]},
            )
            if seo.get("title"):
                yt_title = str(seo["title"]).strip()[:95]
            if seo.get("intro"):
                yt_desc = f"{seo['intro'].strip()}\n\nTimestamps / Chapters:\n{chapter_block}\n\nSubscribe to @RazonGamerz!\n#GTA6 #GamingHighlights #RazonGamerz"
        except Exception:
            pass

        slot_iso = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=2)).replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"[WEEKLY COMPILER] Compiled {len(items)} clips ({probe_duration(final_mp4):.1f}s) -> Title: {yt_title}")

        if args.schedule:
            vid_id, vid_url = upload_and_schedule_short(
                video_path=final_mp4, thumb_path=final_thumb, title=yt_title, description=yt_desc,
                tags=["GTA 6", "Gaming Highlights", "Weekly Gaming Clips", "Twitch Highlights", "RazonGamerz"],
                publish_at_iso=slot_iso,
            )
            watch_url = f"https://www.youtube.com/watch?v={vid_id}"
            print(f"[WEEKLY COMPILER] Scheduled Long-Form Compilation on YouTube Studio: {watch_url} (publishAt={slot_iso})")
            try:
                st = load_cloud_state()
                st["latest_compilation"] = {"video_id": vid_id, "url": watch_url, "title": yt_title}
                save_cloud_state(st)
            except Exception:
                pass
            try:
                drive_upload_file(final_mp4, mime_type="video/mp4", description=yt_title)
            except Exception:
                pass
    finally:
        shutil.rmtree(work, ignore_errors=True)
