"""Background download queue with N concurrent workers.

Jobs are opaque async callables — the bot layer enqueues `lambda: process(...)`
and awaits their completion via the returned `asyncio.Future`. Workers pull
strict-FIFO; the worker count is fixed at construction.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Generic, TypeVar

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
        # Futures still waiting for a free worker, in strict FIFO order.
        # A job is appended on submit and removed when a worker pulls it
        # off the queue. `position()` indexes into this list.
        self._pending: list[asyncio.Future[T]] = []
        # Fires whenever `_pending` changes (job submitted or pulled), so
        # position trackers refresh exactly on change instead of polling.
        self._change: asyncio.Event = asyncio.Event()
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
        self._pending.append(fut)
        self._signal_change()
        return fut

    def _signal_change(self) -> None:
        """Wake everything waiting on a queue change, then install a fresh
        event so the next waiter starts clean."""
        old, self._change = self._change, asyncio.Event()
        old.set()

    @property
    def change_event(self) -> asyncio.Event:
        """The current change-event. Snapshot this *before* reading queue
        state; awaiting it then catches any change that lands afterwards
        (the event is already set if a change raced in between)."""
        return self._change

    @property
    def concurrency(self) -> int:
        """Number of worker slots — the most jobs that can run at once."""
        return self._concurrency

    @property
    def pending(self) -> int:
        """Number of jobs still waiting for a worker (not yet running).

        A job cancelled while mid-queue keeps its slot until a worker
        drains up to it, so this can briefly over-count by the number of
        such cancelled jobs. Fine for the cosmetic position label."""
        return len(self._pending)

    def position(self, fut: asyncio.Future[T]) -> int:
        """0-based index of `fut` among the jobs still waiting for a
        worker. Returns -1 once the job has been picked up (or if it was
        never queued here)."""
        try:
            return self._pending.index(fut)
        except ValueError:
            return -1

    async def _worker(self, idx: int) -> None:
        while self._running:
            try:
                job, fut = await self._queue.get()
            except asyncio.CancelledError:
                return
            try:
                self._pending.remove(fut)
            except ValueError:
                pass
            self._signal_change()
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
