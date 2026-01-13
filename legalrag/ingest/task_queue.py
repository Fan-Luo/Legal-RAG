from __future__ import annotations

from dataclasses import dataclass, field
from queue import Queue, Empty
from threading import Event, Thread
from typing import Callable, Tuple

from legalrag.utils.logger import get_logger

logger = get_logger(__name__)


Task = Tuple[Callable, tuple, dict]


@dataclass
class TaskQueue:
    name: str
    maxsize: int = 0
    _queue: Queue = field(init=False)
    _stop: Event = field(init=False)
    _worker: Thread = field(init=False)

    def __post_init__(self) -> None:
        self._queue = Queue(maxsize=self.maxsize)
        self._stop = Event()
        self._worker = Thread(target=self._run, name=self.name, daemon=True)
        self._worker.start()

    def enqueue(self, fn: Callable, *args, **kwargs) -> None:
        self._queue.put((fn, args, kwargs))

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                fn, args, kwargs = self._queue.get(timeout=0.2)
            except Empty:
                continue
            try:
                fn(*args, **kwargs)
            except Exception as e:
                logger.exception("task failed: %s", e)
            finally:
                self._queue.task_done()

    def stop(self) -> None:
        self._stop.set()
