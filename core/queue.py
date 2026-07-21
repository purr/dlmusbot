"""Background download queue with N concurrent workers.

Jobs are opaque async callables — the bot layer enqueues `lambda: process(...)`
and awaits their completion via the returned `asyncio.Future`.

Fair scheduling: jobs are bucketed by submitter key (`submit(job, key=user)`)
and workers pull round-robin across buckets — one user pasting 100 tracks no
longer delays another user's single request by the whole backlog. Jobs sharing
a key (and all keyless jobs) stay FIFO among themselves; with a single bucket
the queue degenerates to plain FIFO, matching the old behaviour.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict, deque
from typing import Awaitable, Callable, Generic, Optional, TypeVar

T = TypeVar("T")
log = logging.getLogger(__name__)


class DownloadQueue(Generic[T]):
    def __init__(self, concurrency: int = 1):
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self._concurrency = concurrency
        # key -> FIFO of (job, fut). Key order IS the round-robin rotation
        # order: workers pop from the first bucket, then move it to the
        # back. Emptied buckets are dropped so rotation only visits keys
        # with work.
        self._buckets: "OrderedDict[object, deque[tuple[Callable[[], Awaitable[T]], asyncio.Future[T]]]]" = (
            OrderedDict()
        )
        # Counts submitted-but-not-picked-up jobs; workers block on it.
        self._items = asyncio.Semaphore(0)
        # Fires whenever the pending set changes (job submitted or pulled),
        # so position trackers refresh exactly on change instead of polling.
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

    def submit(
        self, job: Callable[[], Awaitable[T]], key: object = None
    ) -> asyncio.Future[T]:
        """Queue `job` under `key` (e.g. the requesting user id). Jobs with
        the same key run FIFO relative to each other; different keys are
        drained round-robin."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[T] = loop.create_future()
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = deque()
            self._buckets[key] = bucket
        bucket.append((job, fut))
        self._items.release()
        self._signal_change()
        return fut

    def _pop_next(
        self,
    ) -> Optional[tuple[Callable[[], Awaitable[T]], asyncio.Future[T]]]:
        """Take the next job round-robin: head of the first bucket, which
        then rotates to the back (or is dropped when emptied)."""
        while self._buckets:
            key = next(iter(self._buckets))
            bucket = self._buckets[key]
            if not bucket:
                del self._buckets[key]
                continue
            item = bucket.popleft()
            if bucket:
                self._buckets.move_to_end(key)
            else:
                del self._buckets[key]
            return item
        return None

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
        return sum(len(b) for b in self._buckets.values())

    def _pending_in_order(self) -> list[asyncio.Future[T]]:
        """Futures in projected dequeue order under round-robin draining:
        depth 0 of every bucket in rotation order, then depth 1, and so
        on. Exact for the cosmetic position label as long as no new
        submits land in between (a submit fires the change event and the
        tracker recomputes)."""
        buckets = list(self._buckets.values())
        out: list[asyncio.Future[T]] = []
        depth = 0
        found = True
        while found:
            found = False
            for b in buckets:
                if depth < len(b):
                    out.append(b[depth][1])
                    found = True
            depth += 1
        return out

    def position(self, fut: asyncio.Future[T]) -> int:
        """0-based index of `fut` in the projected dequeue order among the
        jobs still waiting for a worker. Returns -1 once the job has been
        picked up (or if it was never queued here)."""
        try:
            return self._pending_in_order().index(fut)
        except ValueError:
            return -1

    async def _worker(self, idx: int) -> None:
        while self._running:
            try:
                await self._items.acquire()
            except asyncio.CancelledError:
                return
            got = self._pop_next()
            if got is None:
                continue
            job, fut = got
            self._signal_change()
            if fut.cancelled():
                continue
            try:
                result = await job()
                if not fut.done():
                    fut.set_result(result)
            except asyncio.CancelledError:
                if not fut.done():
                    fut.cancel()
                raise
            except Exception as e:
                log.exception("download worker %d job failed", idx)
                if not fut.done():
                    fut.set_exception(e)
