"""The ``petals`` command: a single entry point that dispatches to subcommands.

    petals up <model> [--join ...]   Start/join a private swarm in one command (recommended)
    petals server <model> ...        The full server with every knob (petals.cli.run_server)
    petals dht ...                   A standalone lightweight DHT bootstrap peer

Each subcommand owns its own argument parsing; this shim just strips the subcommand
name and delegates. Also runnable as ``python -m petals.cli``.
"""

import sys

_COMMANDS = ("up", "server", "dht")

_USAGE = """usage: petals <command> [options]

commands:
  up        Start or join a private swarm in one command (recommended)
              first machine:  petals up <model>
              other machines: petals up <model> --join drift://<peer_id>@<host>:<port>
  server    Run a server with the full set of options (advanced)
  dht       Run a standalone DHT bootstrap peer

Run `petals <command> --help` for command-specific options.
"""


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "help"):
        sys.stdout.write(_USAGE)
        return 0

    command, rest = argv[0], argv[1:]
    if command not in _COMMANDS:
        sys.stderr.write(f"petals: unknown command {command!r}\n\n{_USAGE}")
        return 2

    # Give the delegated parser a clean argv with a sensible prog name in its --help.
    sys.argv = [f"petals {command}", *rest]
    if command == "up":
        from petals.cli.run_up import main as run
    elif command == "server":
        from petals.cli.run_server import main as run
    else:  # dht
        from petals.cli.run_dht import main as run

    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
