# dlmusbot

Telegram music download bot with inline search and direct-link downloads.

## Features

- Inline mode across configured providers (`@your_bot query` or URL).
- Direct chat mode (send track/album/playlist/artist links in DM, share
  shortlinks included). A link from a service we don't support gets an
  explicit reply instead of silence.
- Providers:
  - Spotify (requires `SP_DC`)
  - SoundCloud (auto client_id)
  - YouTube Music (optional cookies)
- Best-effort metadata embedding (title, artist, album, cover, URLs).
- File ID cache for fast repeat deliveries.
- Graceful fallbacks (upload retry, optional ffmpeg paths, non-fatal tagging failures).

## Requirements

- Python 3.12+ recommended
- Optional but recommended: `ffmpeg` and `ffprobe` on `PATH`
- Nothing else to install by hand: if the YouTube bot-gate is ever hit, the
  headless chromium that mints the proof-of-origin token downloads itself in
  the background (system libraries too, on a root Linux host). The token is
  bound to the video *and* to this machine's IP, so it is always minted on the
  host that downloads. Age-restricted videos are a separate wall that only
  `YT_COOKIES_FILE` (a signed-in adult account) can pass.

Install deps in your active virtual environment:

```bash
pip install -r requirements.txt
```

## Configuration

Copy `config.example.py` to `config.py` and fill values.

Key settings:

- `BOT_TOKEN` (required)
- `SP_DC` (required for Spotify)
- `YT_COOKIES_FILE` (optional; recommended path: `data/cookies.youtube.txt`,
  Netscape format; validated at startup and re-read whenever the file changes)
- `YT_BROWSER_POTOKEN` (optional, default on; mints a YouTube proof-of-origin
  token in a self-installing headless chromium when YouTube answers "Sign in to
  confirm you're not a bot", then retries that download once)
- `FORWARD_LOG_CHANNEL_ID` (optional forwarding log channel)
- `DOWNLOAD_CONCURRENCY` (optional; 0 = auto, one worker per CPU core)
- `MAX_FILE_MB`
- `INLINE_RESULTS`
- `SEARCH_PER_PROVIDER`
- `INLINE_SEARCH_PROVIDERS`
- `INLINE_CACHE_SECONDS`

## Run

```bash
python main.py
```

## Notes

- `config.py` is git-ignored; keep secrets there.
- Cached Telegram file IDs are stored in `data/cache.json`.
- Audio temp files are created during jobs and removed after delivery.

