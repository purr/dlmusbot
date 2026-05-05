"""Background download queue with N concurrent workers.

Jobs are opaque async callables — the bot layer enqueues `lambda: process(...)`
and awaits their completion via the returned `asyncio.Future`. Workers pull
strict-FIFO; concurrency cap is set in config.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Generic, Optional, TypeVar

T = TypeVar("T")
log = logging.getLogger(__name__)


class DownloadQueue(Generic[T]):
    def __init__(self, concurrency: int = 1):
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self._concurrency = concurrency
        self._queue: asyncio.Queue[
            tuple[Callable[[], Awaitable[T]], asyncio.Future[T]]
        ] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._workers = [
            asyncio.create_task(self._worker(i), name=f"dlworker-{i}")
            for i in range(self._concurrency)
        ]

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []

    def submit(self, job: Callable[[], Awaitable[T]]) -> asyncio.Future[T]:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[T] = loop.create_future()
        self._queue.put_nowait((job, fut))
        return fut

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    async def _worker(self, idx: int) -> None:
        while self._running:
            try:
                job, fut = await self._queue.get()
            except asyncio.CancelledError:
                return
            if fut.cancelled():
                self._queue.task_done()
                continue
            try:
                result = await job()
                if not fut.done():
                    fut.set_result(result)
            except asyncio.CancelledError:
                if not fut.done():
                    fut.cancel()
                self._queue.task_done()
                raise
            except Exception as e:
                log.exception("download worker %d job failed", idx)
                if not fut.done():
                    fut.set_exception(e)
            finally:
                self._queue.task_done()
