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


# --- Cloud backup (cache + stats -> Telegram channel) ----------------------

# Fully automatic. cache.json + bot_stats.json are continuously backed up to
# a private Telegram channel and MERGED back in on every startup, so the data
# survives a server switch or a wiped disk with zero manual steps. It turns
# itself on as soon as a usable channel exists (this one, or a dedicated one
# below) — no enable flag.
#
# Requirements for the channel:
#   - PRIVATE (never public — the backup contains user IDs).
#   - This bot is an ADMIN with "Post Messages" AND "Edit Messages" rights
#     (channels require "Edit Messages" to pin; the normal pin right does not
#     apply to channels). If the right is missing the backup quietly disables
#     itself and logs why — nothing breaks.
#   - Nothing else gets pinned there (the bot finds its backup via the pin).
#
# Only cache.json + stats are ever uploaded — never tokens/cookies/sp_dc.
# Empty -> reuse FORWARD_LOG_CHANNEL_ID. Both empty -> backup off.
BACKUP_CHANNEL_ID: str = ""
BACKUP_INTERVAL_MINUTES: int = 10
# How many recent backup archives to retain in the channel (older ones are
# pruned only after a new one is safely uploaded + pinned).
BACKUP_KEEP: int = 3
BACKUP_STATE_FILE: str = "data/backup_state.json"


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

# Concurrent downloads. 0 = automatic (one worker per CPU core). Set a
# positive number to override and pin the pool regardless of host.
DOWNLOAD_CONCURRENCY: int = 0

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
