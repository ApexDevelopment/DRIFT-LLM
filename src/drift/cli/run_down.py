"""``drift down``: stop DRIFT-LLM servers running on this machine.

Finds the servers this machine started with ``drift up`` / ``drift server`` (recorded under
``~/.cache/drift/run/``), stops them, and cleans up records left behind by servers that already
exited. By default each server is asked to shut down gracefully -- so it announces itself offline
and takes its p2pd daemon with it -- escalating to a forceful kill if it does not exit in time.

``drift down`` only affects *this* machine: a distributed swarm has no owner that could stop the
other peers. A server kept alive by an external supervisor (a Windows Scheduled Task restart loop,
systemd, a launchd ``KeepAlive`` agent) will just be restarted by that supervisor -- stop it there.
"""

import argparse
import time
from dataclasses import dataclass, field
from typing import List, Optional

from hivemind.utils.logging import get_logger, use_hivemind_log_handler

from drift.utils.server_registry import (
    ServerRecord,
    iter_records,
    process_alive,
    process_command_line,
    terminate_process,
    unregister_server,
)

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)

# If a recorded pid is alive but its command line clearly isn't a drift/python process, the pid was
# almost certainly recycled after a hard kill; we must not terminate an unrelated process.
_DRIFT_MARKERS = ("drift", "run_server", "python")

# Time to wait after a forceful kill before giving up on a process.
_FORCE_GRACE_SECONDS = 5.0


@dataclass
class Classified:
    live: List[ServerRecord] = field(default_factory=list)
    stale: List[ServerRecord] = field(default_factory=list)  # pid gone
    foreign: List[ServerRecord] = field(default_factory=list)  # pid reused by something else


def select_records(records: List[ServerRecord], *, pid: Optional[int], dht_prefix: Optional[str]) -> List[ServerRecord]:
    return [r for r in records if (pid is None or r.pid == pid) and (dht_prefix is None or r.dht_prefix == dht_prefix)]


def classify(records: List[ServerRecord]) -> Classified:
    result = Classified()
    for record in records:
        if not process_alive(record.pid):
            result.stale.append(record)
            continue
        cmdline = process_command_line(record.pid)
        if cmdline is not None and not any(marker in cmdline.lower() for marker in _DRIFT_MARKERS):
            result.foreign.append(record)
        else:
            result.live.append(record)
    return result


def describe(record: ServerRecord) -> str:
    return f"pid {record.pid} ({record.dht_prefix or record.model or 'unknown model'})"


def stop_servers(records: List[ServerRecord], *, force: bool, timeout: float) -> List[ServerRecord]:
    """Stop the given (live) servers and return the records that were confirmed stopped."""
    for record in records:
        logger.info(f"Stopping {describe(record)}{' (force)' if force else ''}")
        terminate_process(record.pid, force=force)

    remaining = _wait_for_exit(records, timeout)
    if remaining and not force:
        for record in remaining:
            logger.warning(f"{describe(record)} did not exit within {timeout:.0f}s; forcing.")
            terminate_process(record.pid, force=True)
        remaining = _wait_for_exit(remaining, _FORCE_GRACE_SECONDS)

    stopped = [r for r in records if r not in remaining]
    for record in stopped:
        unregister_server(record.pid)
    return stopped


def _wait_for_exit(records: List[ServerRecord], timeout: float) -> List[ServerRecord]:
    deadline = time.monotonic() + timeout
    remaining = list(records)
    while remaining and time.monotonic() < deadline:
        time.sleep(0.2)
        remaining = [r for r in remaining if process_alive(r.pid)]
    return remaining


def main():
    parser = argparse.ArgumentParser(
        prog="drift down",
        description="Stop DRIFT-LLM servers running on this machine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pid", type=int, default=None, help="Only act on the server with this pid")
    parser.add_argument("--dht_prefix", default=None, help="Only act on servers announced under this DHT prefix")
    parser.add_argument("--list", action="store_true", help="List running servers without stopping anything")
    parser.add_argument("--force", "-f", action="store_true", help="Kill immediately instead of asking nicely first")
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="Seconds to wait for a graceful shutdown before forcing"
    )
    args = parser.parse_args()

    records = select_records(iter_records(), pid=args.pid, dht_prefix=args.dht_prefix)
    if not records:
        logger.info("No DRIFT-LLM servers are recorded as running on this machine.")
        return

    groups = classify(records)
    for record in groups.stale:
        unregister_server(record.pid)
        logger.info(f"{describe(record)} already stopped; cleaned up its stale record.")
    for record in groups.foreign:
        unregister_server(record.pid)
        logger.warning(
            f"{describe(record)} points at pid {record.pid}, which no longer looks like a drift server "
            f"(pid reuse?); leaving that process alone and removing the record."
        )

    if args.list:
        if groups.live:
            logger.info(f"{len(groups.live)} DRIFT-LLM server(s) running on this machine:")
            for record in groups.live:
                addr = record.maddrs[0] if record.maddrs else "no address"
                logger.info(f"  {describe(record)}  up {int(record.age_seconds)}s  {addr}")
        else:
            logger.info("No DRIFT-LLM servers are running on this machine.")
        return

    if not groups.live:
        logger.info("No running DRIFT-LLM servers to stop.")
        return

    stopped = stop_servers(groups.live, force=args.force, timeout=args.timeout)
    logger.info(f"Stopped {len(stopped)} server(s).")
    still_running = [r for r in groups.live if r not in stopped]
    if still_running:
        logger.error(f"Failed to stop: {', '.join(describe(r) for r in still_running)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
