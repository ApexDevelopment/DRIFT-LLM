"""
Offline tests for the server startup guard: the thread-stack dump helper, the readiness timeout
in ``ModuleContainer.run_in_background`` (which must dump every thread's stack and raise instead
of hanging forever), and the CLI flags that arm them (``--ready_timeout``, ``--debug_hang_dump``).
No swarm / network / download required (does not import test_utils).
"""
import logging
import multiprocessing as mp
import threading
from types import SimpleNamespace

import pytest

from drift.cli.run_server import build_parser
from drift.server.server import ModuleContainer
from drift.utils.misc import format_all_thread_stacks


def _parked_thread_function_for_dump(started: threading.Event, release: threading.Event):
    started.set()
    release.wait(timeout=30)


def test_format_all_thread_stacks_names_threads_and_frames():
    started, release = threading.Event(), threading.Event()
    thread = threading.Thread(
        target=_parked_thread_function_for_dump, args=(started, release), name="parked-probe", daemon=True
    )
    thread.start()
    try:
        assert started.wait(timeout=10)
        dump = format_all_thread_stacks()
    finally:
        release.set()
        thread.join(timeout=10)

    assert 'Thread "MainThread"' in dump
    assert 'Thread "parked-probe"' in dump
    assert "_parked_thread_function_for_dump" in dump  # the parked frame is localizable from the dump


def test_module_container_ready_timeout_dumps_and_raises():
    container = ModuleContainer.__new__(ModuleContainer)  # skip the heavy __init__: no DHT, no blocks
    threading.Thread.__init__(container, daemon=True)
    container.ready_timeout = 0.2
    container.runtime = SimpleNamespace(ready=mp.Event())  # never set -> startup never completes
    container.run = lambda: None  # the real run() needs handlers; the guard under test is in run_in_background

    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    server_logger = logging.getLogger("drift.server.server")
    handler = _Capture()
    server_logger.addHandler(handler)
    try:
        with pytest.raises(TimeoutError, match="didn't notify .ready in 0.2"):
            container.run_in_background(await_ready=True)
    finally:
        server_logger.removeHandler(handler)

    dump = "\n".join(records)
    assert "did not become ready within 0.2 seconds" in dump
    assert 'Thread "MainThread"' in dump  # the stack dump itself made it into the log


def test_cli_exposes_startup_guard_flags():
    args = vars(build_parser().parse_args(["dummy/model", "--new_swarm"]))
    assert args["ready_timeout"] == 120
    assert args["debug_hang_dump"] is None

    args = vars(
        build_parser().parse_args(["dummy/model", "--new_swarm", "--ready_timeout", "45", "--debug_hang_dump", "30"])
    )
    assert args["ready_timeout"] == 45.0
    assert args["debug_hang_dump"] == 30.0
