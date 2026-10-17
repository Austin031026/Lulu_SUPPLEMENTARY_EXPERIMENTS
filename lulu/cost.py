"""Explicit timing boundaries and cost-only training policy."""
import time
from contextlib import contextmanager


def efficient(args):
    return bool(getattr(args, 'cost_mode', False))


def needs_hindsight(args):
    return not (efficient(args) and args.reasoning_ablation == 'vanilla')


def needs_teacher(args):
    return not (efficient(args) and args.reasoning_ablation == 'opsd')


def synchronize():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def measured(totals, name):
    # Only use inside workers, where CUDA_VISIBLE_DEVICES identifies the role.
    synchronize()
    started = time.monotonic()
    try:
        yield
    finally:
        synchronize()
        totals[name] = totals.get(name, 0.) + time.monotonic() - started


def summarize_workers(rows):
    keys = sorted({key for row in rows for key in row})
    return {key: {'sum_worker_seconds': sum(row.get(key, 0.) for row in rows),
                  'max_worker_seconds': max(row.get(key, 0.) for row in rows)} for key in keys}
