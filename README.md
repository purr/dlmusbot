# dlmusbot

Telegram music download bot with inline search and direct-link downloads.

## Features

- Inline mode across configured providers (`@your_bot query` or URL).
- Direct chat mode (send track/album/playlist links in DM).
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

Install deps in your active virtual environment:

```bash
pip install -r requirements.txt
```

## Configuration

Copy `config.example.py` to `config.py` and fill values.

Key settings:

- `BOT_TOKEN` (required)
- `SP_DC` (required for Spotify)
- `YT_COOKIES_FILE` (optional; recommended path: `data/cookies.youtube.txt`)
- `FORWARD_LOG_CHANNEL_ID` (optional forwarding log channel)
- `DOWNLOAD_CONCURRENCY`
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

