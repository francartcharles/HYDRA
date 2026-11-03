"""
Shared multi-core execution helper.

Four of the app's computational hot paths are, in different ways, an
"embarrassingly parallel" map over independent work items with no shared
mutable state between them:

- Monte-Carlo uncertainty propagation (core/process_shpb.py,
  core/process_di.py, called from ui/pages/page_process.py): one
  independent call per Monte-Carlo draw.
- Wave-dispersion correction (core/dispersion.py): the incident/reflected/
  transmitted Pochhammer-Chree (or viscoelastic) branch solves are each
  independent of the other two -- each branch's own frequency-by-frequency
  solve IS sequential internally (it seeds every step from the previous
  one, a continuation method), so only the (up to) 3 branches themselves
  are parallelized, never the frequencies within one branch.
- DIC (image correlation, core/dic_processing.py): within one frame, every
  mesh point's subset search is independent of every other point's -- the
  frame *sequence* itself stays sequential ("moving" tracking mode centers
  each frame's search on the previous frame's own result, and a point
  dropped by the correlation/neighbor-consistency filters must stay
  dropped in every later frame).
- Bayesian viscoelastic calibration (core/bayes_visco_calibration.py): the
  initial Latin-Hypercube sampling phase evaluates independent candidate
  points before any Gaussian-Process fitting begins; the adaptive search
  loop that follows proposes one point at a time from a GP fit on every
  point evaluated so far, so it is NOT parallelized here -- doing so would
  change the search's own behavior (which point it looks at next), not
  just its speed.

This module gives every one of those call sites the same worker-count knob
(`Project.n_workers`, set on the Test Setup page's "Performance" card)
through one small, process-based utility, instead of four separate ad hoc
implementations.

Processes, not threads: the per-item work in all four cases is dominated
by plain Python (list/loop-heavy numeric code, sympy symbolic solving)
rather than GIL-releasing C calls, so threads alone would not actually run
concurrently on more than one core here.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Callable, Optional

# Per-worker-process global, populated ONCE per worker (by the pool's
# `initializer`, not once per task) -- see `run_map`'s `context` argument.
_CONTEXT = None


class OperationCancelled(Exception):
    """Raised by `check_cancelled` when a long-running computation's
    caller-supplied `cancel_check()` reports the user asked it to stop --
    shared by every UI "Stop" button wired to a background computation
    (DIC in page_image_correlation.py, X-FEM-DIC in
    page_crack_wave_analysis.py, Bayesian viscoelastic calibration in
    page_visco_calib.py; see each page's own `_CalibWorker`/`_DICWorker`/
    `_XfemDicWorker`). Propagates up through the computing core function
    (run_dic, compute_xfem_dic_time_series, calibrate_bayesian) to its
    QThread worker's own `run()`, which catches it SEPARATELY from a
    genuine failure and emits a `cancelled` signal instead of `failed` --
    so the UI can show "stopped by the user", not an error."""


def check_cancelled(cancel_check: Optional[Callable[[], bool]]) -> None:
    """One-line check to drop into a long-running loop's body: raises
    `OperationCancelled` if `cancel_check` is not None and calling it
    returns truthy. `cancel_check` is typically a `threading.Event`'s own
    `.is_set` (cheap, thread-safe to read from the worker thread while the
    UI thread's "Stop" button handler calls `.set()`), so this never
    blocks -- cancellation only actually stops the loop the next time this
    is called, i.e. cooperative, not preemptive: a single frame/iteration/
    fit already in progress always finishes first."""
    if cancel_check is not None and cancel_check():
        raise OperationCancelled("Stopped by user.")


def available_cpu_count() -> int:
    return os.cpu_count() or 1


def _init_worker(context) -> None:
    global _CONTEXT
    _CONTEXT = context


def _call_with_context(func: Callable, item):
    """Runs inside the worker process: `func(context, item)` when a
    context was provided to `run_map`, else `func(item)`."""
    if _CONTEXT is not None:
        return func(_CONTEXT, item)
    return func(item)


def run_map(
    func: Callable,
    items: list,
    n_workers: int = 1,
    context=None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    use_threads: bool = False,
) -> list[tuple[int, object, Optional[BaseException]]]:
    """Applies `func` to every item in `items`, returning a list of
    `(index, result, error)` triples in the SAME ORDER as `items`
    (`result` is `None` exactly when `error` is not `None`).

    `func(item)` -- or `func(context, item)` when `context` is given --
    must be a plain module-level function (picklable BY REFERENCE, so
    never a lambda or a bound method) whenever `n_workers > 1` and
    `use_threads` is False.

    `context` (optional): data shared READ-ONLY across every item (bars,
    specimen, physical constants...). With processes (the default),
    pickled ONCE PER WORKER PROCESS via the pool's `initializer`, never
    once per item -- cheap even when `items` is long and `context` itself
    is large (e.g. a bar's whole raw signal), unlike passing the same
    large object as part of every task. With threads, `context` is simply
    shared in memory (no pickling at all), same as any other closed-over
    object.

    `progress_callback(done, total)` is called from THIS (the calling)
    process/thread only, right after each item completes -- always safe to
    update a Qt progress bar from it directly, unlike `func` itself when
    `n_workers > 1` and `use_threads` is False (which runs in a different
    process entirely and can't touch GUI objects).

    `use_threads=True` uses a `ThreadPoolExecutor` instead of a
    `ProcessPoolExecutor` -- only worth it when `func`'s actual hot loop
    is a C call that releases the GIL (e.g. OpenCV's `matchTemplate`, used
    by core/dic_processing.py); for the plain-Python-bound work in the
    other call sites (Monte-Carlo draws, sympy root-solving), threads
    would not run concurrently at all and processes are required for a
    real speedup. Avoids the picklability requirement (and the per-worker
    context pickling cost) entirely, since threads share memory directly.

    `n_workers <= 1` runs a plain serial loop IN THIS PROCESS/THREAD,
    calling `func` directly -- identical behavior/error handling to a
    hand-written `for` loop, and the only path that doesn't require
    `func`/`items`/`context` to be picklable even without `use_threads`.
    This is the default, so nothing changes unless a project explicitly
    opts into more than one worker."""
    n = len(items)
    results: list = [None] * n

    if n_workers <= 1:
        for i, item in enumerate(items):
            try:
                value = func(context, item) if context is not None else func(item)
                results[i] = (i, value, None)
            except BaseException as exc:  # noqa: BLE001 - reported per item, matching the loops this replaces
                results[i] = (i, None, exc)
            if progress_callback:
                progress_callback(i + 1, n)
        return results

    if use_threads:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            if context is not None:
                futures = {pool.submit(func, context, item): i for i, item in enumerate(items)}
            else:
                futures = {pool.submit(func, item): i for i, item in enumerate(items)}
            done = 0
            for future in as_completed(futures):
                i = futures[future]
                try:
                    results[i] = (i, future.result(), None)
                except BaseException as exc:  # noqa: BLE001
                    results[i] = (i, None, exc)
                done += 1
                if progress_callback:
                    progress_callback(done, n)
        return results

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker, initargs=(context,)) as pool:
        futures = {pool.submit(_call_with_context, func, item): i for i, item in enumerate(items)}
        done = 0
        for future in as_completed(futures):
            i = futures[future]
            try:
                results[i] = (i, future.result(), None)
            except BaseException as exc:  # noqa: BLE001
                results[i] = (i, None, exc)
            done += 1
            if progress_callback:
                progress_callback(done, n)
    return results


def chunk_items(items: list, n_chunks: int) -> list[list]:
    """Splits `items` into up to `n_chunks` roughly-equal contiguous
    chunks (never more chunks than items, never an empty chunk). Used to
    batch many small per-item tasks (e.g. one DIC mesh point) into one
    task per worker, instead of paying a submit/collect round-trip for
    every single point."""
    n_chunks = max(1, min(n_chunks, len(items)))
    if n_chunks <= 1:
        return [items]
    size = -(-len(items) // n_chunks)  # ceil division
    return [items[i:i + size] for i in range(0, len(items), size)]
