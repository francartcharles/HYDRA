"""
Shared QThread worker for running a long calibration/optimization function
off the GUI thread, extracted from `ui/pages/page_visco_calib.py`'s own
`_CalibWorker` (that page's Bayesian viscoelastic-model calibration was the
first place this pattern was needed; `ui/pages/page_bayes_calib.py`'s
PyMC-based Bayesian Calibration run is the second, and duplicating the
class a second time would just be two copies to keep in sync for no
reason -- the pattern itself has nothing viscoelastic-specific about it).

Usage (see either page's `_on_run` for the full wiring):

    self._thread = QThread()
    self._worker = CalibWorker(fn, kwargs, cancel_event=self._cancel_event)
    self._worker.moveToThread(self._thread)
    self._thread.started.connect(self._worker.run)
    self._worker.progress.connect(self._on_progress)
    self._worker.finished.connect(self._on_finished)
    self._worker.failed.connect(self._on_failed)
    self._worker.cancelled.connect(self._on_cancelled)
    self._worker.finished.connect(self._thread.quit)
    self._worker.failed.connect(self._thread.quit)
    self._worker.cancelled.connect(self._thread.quit)
    self._thread.start()

`fn` must accept a `progress_cb(i, total)` keyword argument (silently
ignored if a particular `fn` never calls it) and, when `cancel_event` is
given, a `cancel_check` keyword argument -- a zero-arg callable returning
True once cancellation was requested (matching `core.parallel.
OperationCancelled`'s own cooperative-cancellation convention, which
`fn` is expected to raise once `cancel_check()` returns True).
"""
from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Signal

from core.parallel import OperationCancelled


class CalibWorker(QObject):
    progress = Signal(int, int)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, fn, kwargs, cancel_event: threading.Event | None = None):
        super().__init__()
        self.fn = fn
        self.kwargs = kwargs
        # Only meaningful for an `fn` that actually accepts a
        # `cancel_check` kwarg -- a caller running a fast, non-cancellable
        # `fn` simply passes cancel_event=None, and `cancel_check` is
        # never added to `self.kwargs` below.
        self.cancel_event = cancel_event

    def run(self):
        try:
            kwargs = dict(self.kwargs)
            if self.cancel_event is not None:
                kwargs["cancel_check"] = self.cancel_event.is_set
            result = self.fn(progress_cb=lambda i, total: self.progress.emit(i, total), **kwargs)
            self.finished.emit(result)
        except OperationCancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
