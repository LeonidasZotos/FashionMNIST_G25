import concurrent
import os
import time
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace
from typing import *

from tqdm import tqdm

# A lot of these are inspired by fast.ai code written in part by Jeremy Howard for a deep learning library.
# They were used to make our code more efficient and better and are not directly related to any ML work.


def num_cpus() -> int:
    "Get number of cpus"
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


_default_cpus = min(16, num_cpus())
defaults = SimpleNamespace(
    cpus=_default_cpus, cmap="viridis", return_fig=False, silent=False
)


def ifnone(a, b):
    """
    Return if None
    """
    return b if a is None else a


def parallel(func, arr: Collection, max_workers: int = 8, leave=False):  #%t
    "Call `func` on every element of `arr` in parallel using `max_workers`."
    max_workers = ifnone(max_workers, defaults.cpus)
    if max_workers < 2:
        results = [func(o) for i, o in tqdm(enumerate(arr), total=len(arr))]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(func, o) for i, o in enumerate(arr)]
            results = []
            for f in tqdm(concurrent.futures.as_completed(futures), total=len(arr)):
                results.append(f.result())
    if any([o is not None for o in results]):
        return results
