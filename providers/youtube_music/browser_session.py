"""Browser-minted YouTube proof-of-origin tokens.

YouTube meets server IPs with "Sign in to confirm you're not a bot" unless
the request carries a proof-of-origin (PO) token, and only YouTube's own
JavaScript can mint one. yt-dlp cannot generate it; every working solution
lets a real browser do the minting and hands the result over — bgutil runs
BotGuard under Node, getpot-wpc drives a local Chrome. This does the same
job with Playwright's own chromium, so there's no second service to run
and no Node on the box.

Nothing here needs a manual install: the first time a gate is hit and the
browser turns out to be missing, chromium is downloaded in the background
(and on a root Linux host, its system libraries too).

Two properties decide the design:

  * The token is bound to the video id — yt-dlp logs "Detected experiment
    to bind GVS PO Token to video ID for web client" — so it is minted per
    download, never cached across videos.
  * It is bound to the IP that minted it, so this only helps when it runs
    on the same host that downloads.

On a host YouTube isn't gating, the browser is served plain media URLs with
no token in them at all (verified: not one `pot` parameter on a clean
residential IP). `mint()` then returns None and logs why. That is the
expected result there, not a malfunction — the caller simply carries on
with the failure it already had.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import parse_qs, urlparse

from playwright.async_api import async_playwright

log = logging.getLogger(__name__)

# How long to sit on the watch page waiting for the player to fire a media
# request. The token rides on the first one; if none arrives by then the
# page is either still negotiating or YouTube isn't asking for a token.
_MINT_TIMEOUT_S = 25.0

# Chromium is ~115 MB and `install-deps` runs apt; both are slow on a small
# VPS, and neither is worth abandoning halfway.
_INSTALL_TIMEOUT_S = 900.0

# Playback only starts on its own with these; a paused player never
# requests media, and never reveals a token.
_LAUNCH_ARGS = ["--autoplay-policy=no-user-gesture-required", "--mute-audio"]

# Playwright's way of saying "the browser isn't downloaded yet", plus the
# loader error you get when chromium is present but its system libs aren't.
_INSTALLABLE_MARKERS = (
    "executable doesn't exist",
    "playwright install",
    "error while loading shared libraries",
    "host system is missing dependencies",
)


@dataclass(frozen=True)
class PoToken:
    """A GVS PO token and the visitor identity it is bound to. Both have to
    be handed to yt-dlp together — the token is meaningless against a
    different visitor_data."""

    token: str
    visitor_data: str
    video_id: str


def _is_root_linux() -> bool:
    return os.name != "nt" and hasattr(os, "geteuid") and os.geteuid() == 0


def _launch_args() -> list[str]:
    args = list(_LAUNCH_ARGS)
    # Chromium refuses to sandbox when it runs as uid 0, which is exactly
    # how a pm2-managed bot on a VPS runs. Drop the sandbox only in that
    # case — never on a normal user account, where it's a real boundary.
    if _is_root_linux():
        args.append("--no-sandbox")
    return args


def _looks_installable(err: str) -> bool:
    e = err.lower()
    return any(m in e for m in _INSTALLABLE_MARKERS)


class BrowserTokenMinter:
    """Mints PO tokens on demand from a headless chromium.

    One browser is launched on first use and reused; mints are serialised
    so a burst of gated downloads doesn't open a dozen chromium tabs at
    once. A missing browser installs itself once, in the background.
    """

    def __init__(self, *, timeout_s: float = _MINT_TIMEOUT_S) -> None:
        self._timeout_s = timeout_s
        self._lock = asyncio.Lock()
        self._pw = None
        self._browser = None
        self._disabled = False
        self._install_task: Optional[asyncio.Task] = None
        self._install_proc: Optional[asyncio.subprocess.Process] = None
        self._install_attempted = False

    # ---- browser lifecycle ------------------------------------------------

    async def _browser_or_none(self):
        if self._browser is not None:
            return self._browser
        if self._install_task is not None and not self._install_task.done():
            log.debug("chromium install still running; skipping this mint")
            return None
        try:
            self._pw = await async_playwright().start()
            self._browser = await self._pw.chromium.launch(
                headless=True, args=_launch_args()
            )
        except Exception as e:
            await self._shutdown()
            if not self._install_attempted and _looks_installable(str(e)):
                self._start_install()
                return None
            log.error(
                "PO token minter disabled - chromium won't launch (%s: %s)",
                type(e).__name__,
                e,
            )
            self._disabled = True
            return None
        return self._browser

    # ---- self-install -----------------------------------------------------

    def _start_install(self) -> None:
        """Fetch chromium in the background.

        The download that got us here is already failing and the browser is
        ~115 MB, so making that request sit through the install would turn
        one bad download into a queue slot hung for minutes. Start the
        install, let this request fall back to another provider as it
        already would, and pick the browser up on the next gate."""
        self._install_attempted = True
        self._install_task = asyncio.create_task(self._install())

    async def _install(self) -> None:
        started = time.monotonic()
        log.warning(
            "chromium is missing - installing it in the background (~115 MB). "
            "YouTube bot-gate retries start working once this finishes; "
            "downloads keep falling back to the other providers meanwhile"
        )
        # Chromium's system libraries: apt-only, root-only, linux-only. A
        # non-root host that needs them fails at launch instead, and says so.
        if _is_root_linux() and not await self._run_playwright("install-deps", "chromium"):
            log.warning(
                "playwright install-deps failed - continuing anyway; chromium "
                "may still launch if the libraries are already present"
            )
        if await self._run_playwright("install", "chromium"):
            log.info("chromium installed in %.0fs", time.monotonic() - started)
            return
        self._disabled = True
        log.error(
            "chromium install failed - PO token minting is off for this run. "
            "Retry by hand with: %s -m playwright install chromium",
            sys.executable,
        )

    async def _run_playwright(self, *args: str) -> bool:
        try:
            self._install_proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "playwright",
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            out, _ = await asyncio.wait_for(
                self._install_proc.communicate(), timeout=_INSTALL_TIMEOUT_S
            )
        except Exception as e:
            log.error(
                "playwright %s failed (%s: %s)", " ".join(args), type(e).__name__, e
            )
            return False
        finally:
            proc, self._install_proc = self._install_proc, None
        if proc is not None and proc.returncode != 0:
            tail = (out or b"").decode("utf-8", "replace").strip()[-400:]
            log.error(
                "playwright %s exited %s: %s", " ".join(args), proc.returncode, tail
            )
            return False
        return True

    # ---- minting ----------------------------------------------------------

    async def mint(self, video_id: str) -> Optional[PoToken]:
        """Load `video_id` in a real browser and harvest the PO token its
        player attaches to the media request. None when the browser was
        served no token, which means YouTube isn't gating this host."""
        if self._disabled:
            return None
        async with self._lock:
            browser = await self._browser_or_none()
            if browser is None:
                return None
            try:
                return await asyncio.wait_for(
                    self._mint(browser, video_id), timeout=self._timeout_s + 15
                )
            except asyncio.TimeoutError:
                log.warning("PO token mint for %s timed out", video_id)
                return None
            except Exception as e:
                log.warning(
                    "PO token mint for %s failed (%s: %s)",
                    video_id,
                    type(e).__name__,
                    e,
                )
                return None

    async def _mint(self, browser, video_id: str) -> Optional[PoToken]:
        context = await browser.new_context()
        try:
            page = await context.new_page()
            found: dict[str, str] = {}

            def on_request(req) -> None:
                if "videoplayback" not in req.url or "pot" in found:
                    return
                pot = parse_qs(urlparse(req.url).query).get("pot")
                if pot:
                    found["pot"] = pot[0]

            page.on("request", on_request)
            await page.goto(
                f"https://www.youtube.com/watch?v={video_id}",
                wait_until="domcontentloaded",
                timeout=self._timeout_s * 1000,
            )
            loop = asyncio.get_running_loop()
            deadline = loop.time() + self._timeout_s
            while "pot" not in found and loop.time() < deadline:
                await asyncio.sleep(0.25)
            if "pot" not in found:
                log.info(
                    "no PO token offered for %s - YouTube is not gating this "
                    "host, so the original failure stands",
                    video_id,
                )
                return None
            visitor_data = await page.evaluate(
                "() => (window.ytcfg && ytcfg.get && ytcfg.get('VISITOR_DATA')) || ''"
            )
            if not visitor_data:
                # The token is only valid against the visitor it was minted
                # for; without that half it's unusable, so say so rather
                # than hand yt-dlp something that silently won't work.
                log.warning("PO token for %s has no visitor_data; discarding", video_id)
                return None
            log.info("minted PO token for %s", video_id)
            return PoToken(
                token=found["pot"], visitor_data=visitor_data, video_id=video_id
            )
        finally:
            await context.close()

    # ---- shutdown ---------------------------------------------------------

    async def _shutdown(self) -> None:
        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception as e:
                log.debug("browser close failed (%s: %s)", type(e).__name__, e)
            self._browser = None
        if self._pw is not None:
            try:
                await self._pw.stop()
            except Exception as e:
                log.debug("playwright stop failed (%s: %s)", type(e).__name__, e)
            self._pw = None

    async def close(self) -> None:
        # Kill a half-finished install rather than leaving an orphaned
        # download running after the bot exits.
        proc = self._install_proc
        if proc is not None and proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        if self._install_task is not None:
            self._install_task.cancel()
            try:
                await self._install_task
            except (asyncio.CancelledError, Exception):
                pass
            self._install_task = None
        async with self._lock:
            await self._shutdown()
