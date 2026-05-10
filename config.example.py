"""Local configuration template. Copy to `config.py` and fill in real values.

`config.py` is git-ignored so secrets never leave the machine.
No environment variables are used — everything lives here.
"""

# --- Telegram ---------------------------------------------------------------

# BotFather token. Required.
BOT_TOKEN: str = "123456:REPLACE_ME"

# Optional channel where each successful delivery is forwarded for logging.
# The bot also posts a "Requested by ..." attribution line as a reply.
# Supports numeric IDs ("-100...") or public usernames ("@channel_name").
# Empty = disabled.
FORWARD_LOG_CHANNEL_ID: str = ""


# --- Spotify ---------------------------------------------------------------

# `sp_dc` cookie value from open.spotify.com. Required for any Spotify use
# (search + download). Get it from the browser dev tools after logging in.
SP_DC: str = ""


# --- SoundCloud ------------------------------------------------------------

# Public web client_id is auto-extracted from soundcloud.com on cold start
# and refreshed automatically on 401. No config required.


# --- YouTube Music ---------------------------------------------------------

# Optional cookie file path for age-gated content or your private uploads.
# Recommended location: "data/cookies.youtube.txt"
# Empty = no cookies (works for ~all public Music tracks via the Android
# Music client trick). Cobalt.tools and Invidious use the same approach.
YT_COOKIES_FILE: str = ""


# --- Download / queue ------------------------------------------------------

# Concurrent downloads through the global pool. Higher = faster bursts but
# more memory. 3 is a good default for Telegram bots.
DOWNLOAD_CONCURRENCY: int = 3

# Hard cap on output file size in MB. Telegram bot limit is 50 MB without
# Premium, 2000 MB with. Files above this are rejected with a friendly note.
MAX_FILE_MB: int = 50

# JSON cache mapping (provider, track_id, format) -> Telegram file_id. This
# is just metadata — no audio bytes are stored locally. Survives restarts so
# repeat downloads return instantly via Telegram's CDN.
CACHE_FILE: str = "data/cache.json"

# JSON file backing /stats — one event per successful delivery, pruned to
# the last 366 days so the rolling-year window stays accurate.
STATS_FILE: str = "data/bot_stats.json"

# Audio downloads use a per-job temporary directory created with
# tempfile.TemporaryDirectory. Files are deleted immediately after upload.
# Nothing persists on disk except the JSON cache above.


# --- Search / inline -------------------------------------------------------

# How many results to show in inline mode. Telegram caps at 50.
INLINE_RESULTS: int = 50

# How many results to ask each provider for before fuzzy-merging. Higher gives
# better quality at the cost of latency.
SEARCH_PER_PROVIDER: int = 25

# Providers searched by inline text queries. Order doesn't matter — results
# are merged then sorted by fuzzy match score.
INLINE_SEARCH_PROVIDERS: list[str] = ["spotify", "soundcloud"]

# Telegram inline result cache TTL in seconds.
INLINE_CACHE_SECONDS: int = 60 * 60
