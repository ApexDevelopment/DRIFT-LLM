"""
Offline tests for the server registry and the `drift down` teardown logic (src/drift/utils/
server_registry.py + src/drift/cli/run_down.py). No swarm: a couple of tests spawn a short-lived
python "sleeper" to stand in for a real server process; the rest drive the pure record logic.
"""

import subprocess
import sys
import time

import pytest

from drift.cli import run_down
from drift.utils import server_registry
from drift.utils.server_registry import (
    ServerRecord,
    iter_records,
    process_alive,
    register_server,
    terminate_process,
    unregister_server,
)

ALIVE_DEADLINE = 10.0


@pytest.fixture(autouse=True)
def isolated_run_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(server_registry, "RUN_DIR", tmp_path / "run")


def _spawn_sleeper() -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])


def _wait_until_dead(pid: int) -> None:
    deadline = time.monotonic() + ALIVE_DEADLINE
    while time.monotonic() < deadline:
        if not process_alive(pid):
            return
        time.sleep(0.1)
    raise AssertionError(f"pid {pid} did not exit")


def test_register_iter_unregister_roundtrip():
    path = register_server(model="tiny/model", dht_prefix="tinyprefix", maddrs=["/ip4/1.2.3.4/tcp/5"])
    assert path.exists()

    records = iter_records()
    assert len(records) == 1
    record = records[0]
    assert record.dht_prefix == "tinyprefix"
    assert record.model == "tiny/model"
    assert record.maddrs == ["/ip4/1.2.3.4/tcp/5"]
    assert record.path == str(path)

    unregister_server()  # defaults to the current pid, which is what register_server recorded
    assert iter_records() == []


def test_iter_records_skips_corrupt_files():
    register_server(model="ok/model", dht_prefix="ok", maddrs=[])
    (server_registry.RUN_DIR / "server-999999.json").write_text("{ not valid json")
    records = iter_records()
    assert [r.dht_prefix for r in records] == ["ok"]


def test_process_alive_and_terminate():
    sleeper = _spawn_sleeper()
    try:
        assert process_alive(sleeper.pid)
        terminate_process(sleeper.pid)
        _wait_until_dead(sleeper.pid)
        assert not process_alive(sleeper.pid)
    finally:
        sleeper.kill()


def test_process_alive_false_for_dead_pid():
    sleeper = _spawn_sleeper()
    sleeper.kill()
    sleeper.wait(timeout=ALIVE_DEADLINE)
    assert not process_alive(sleeper.pid)


def test_select_records_filters():
    records = [
        ServerRecord(pid=1, model="m", dht_prefix="a"),
        ServerRecord(pid=2, model="m", dht_prefix="b"),
    ]
    assert [r.pid for r in run_down.select_records(records, pid=None, dht_prefix="b")] == [2]
    assert [r.pid for r in run_down.select_records(records, pid=1, dht_prefix=None)] == [1]
    assert run_down.select_records(records, pid=3, dht_prefix=None) == []


def test_classify_splits_live_stale_foreign(monkeypatch):
    live = ServerRecord(pid=11, model="m", dht_prefix="live")
    stale = ServerRecord(pid=12, model="m", dht_prefix="stale")
    foreign = ServerRecord(pid=13, model="m", dht_prefix="foreign")

    monkeypatch.setattr(run_down, "process_alive", lambda pid: pid in (11, 13))
    monkeypatch.setattr(
        run_down,
        "process_command_line",
        lambda pid: {11: "python -m drift.cli.run_server", 13: "C:/Windows/notepad.exe"}.get(pid),
    )

    groups = run_down.classify([live, stale, foreign])
    assert [r.pid for r in groups.live] == [11]
    assert [r.pid for r in groups.stale] == [12]
    assert [r.pid for r in groups.foreign] == [13]


def test_stop_servers_terminates_and_cleans_record():
    sleeper = _spawn_sleeper()
    register_server(model="m", dht_prefix="p", maddrs=[])
    # Rewrite the just-written record so its pid is the sleeper, not the test runner.
    record = ServerRecord(pid=sleeper.pid, model="m", dht_prefix="p")
    (server_registry.RUN_DIR / f"server-{sleeper.pid}.json").write_text(
        '{"pid": %d, "model": "m", "dht_prefix": "p", "maddrs": [], "started_at": 0.0}' % sleeper.pid
    )
    unregister_server()  # drop the test-runner's own record from register_server above
    try:
        stopped = run_down.stop_servers([record], force=False, timeout=ALIVE_DEADLINE)
        assert [r.pid for r in stopped] == [sleeper.pid]
        assert not process_alive(sleeper.pid)
        assert not (server_registry.RUN_DIR / f"server-{sleeper.pid}.json").exists()
    finally:
        sleeper.kill()


def test_main_list_does_not_stop(monkeypatch, capsys):
    record = ServerRecord(pid=4242, model="m", dht_prefix="listme")
    monkeypatch.setattr(run_down, "iter_records", lambda: [record])
    monkeypatch.setattr(run_down, "process_alive", lambda pid: True)
    monkeypatch.setattr(run_down, "process_command_line", lambda pid: "python drift")
    terminated = []
    monkeypatch.setattr(run_down, "terminate_process", lambda pid, force=False: terminated.append(pid))
    monkeypatch.setattr(sys, "argv", ["drift down", "--list"])

    run_down.main()
    assert terminated == []
