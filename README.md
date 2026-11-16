# GTA VI Live-Stream & Google Drive -> Automated YouTube Shorts Bot

A 100% cloud-based, stateless automated clipping and YouTube Studio scheduling engine built for the **Grand Theft Auto VI (November 19, 2026)** launch. Reverse-engineered and upgraded from `haziqahmedqasmi/peakcut`.

## What This Repository Does Automatically
1. **Two-Way Google Drive + Live Stream Ingest**:
   - Every 45 minutes, GitHub Actions checks your **Google Drive Inbox Folder** (`GDRIVE_INBOX_FOLDER_ID`) for any new stream recording (`Streamer - Stream Title.mp4`).
   - If the Drive Inbox is empty, it automatically scans top GTA VI live streamers (`kaicenat`, `xqc`, `caseoh_`, `tarik`, `adinross`, `IShowSpeed`, `TypicalGamer`) and captures a 15-minute live slice.
2. **Fast Speech-to-Text + Dual-Signal AI Highlight Scout**:
   - Transcribes word-by-word timestamps using YouTube native `json3` captions, **Groq Cloud Whisper** (`whisper-large-v3-turbo`), or cached **Sherpa-ONNX Parakeet 0.6B**.
   - Scores audio loudness Z-score spikes (`(LOUD)`) and uses **Gemini 2.5 Flash** to select the top 20–55s viral moments.
   - Transliterates Hindi/Urdu words into Roman Hinglish (`roman.py`) so captions never render blank boxes.
3. **TrueType 9:16 ASS Word-Pop Rendering**:
   - Renders 1080x1920 vertical Shorts (`blur`, `zoom`, or `crop` layout) using bundled `Poppins-ExtraBold.ttf` and `Anton-Regular.ttf` fonts, `-14 LUFS` loudness normalization, and background radio music frequency filtering (`--isolate-voice`).
4. **GTA VI Algorithm SEO + 3:2 Safe-Zone Thumbnail + Native `publishAt` Scheduling**:
   - Generates a high-CTR title, 150-word keyword-rich description, and 15 long-tail tags.
   - Composites a custom Shorts thumbnail with the hook locked inside the central 3:2 safe band (`y = 660..1260`).
   - Saves the finished `.mp4`, `_thumb.jpg`, and `_seo.json` to your **Google Drive Output Folder** (`GDRIVE_OUTPUT_FOLDER_ID`) AND uploads directly to **YouTube Studio** scheduled via `status.publishAt`!

## Repository File Structure
```text
gta6-shorts-bot/
├── .github/workflows/clip_and_schedule.yml   # Automated 45-min GitHub Actions workflow
├── fonts/                                    # Bundled Poppins-ExtraBold, Anton & ArchivoBlack TTFs
├── work/clips/                               # Local/CI output directory
├── youtube_seo_scheduler.py                  # Step 1: SEO engine, 3:2 thumbnail composer, Drive & YouTube API
├── stream_clipper.py                         # Step 2: Stream/Drive ingest, ASR, AI scout & 9:16 ASS renderer
├── roman.py                                  # Hindi/Urdu/Arabic/Cyrillic -> Roman Hinglish transliterator
├── colab_youtube_to_drive.ipynb              # 1-Click Google Colab bridge for YouTube-only live streams
├── requirements.txt                          # Lightweight Python dependencies
└── .env.example                              # Template for environment variables / GitHub Secrets
```
