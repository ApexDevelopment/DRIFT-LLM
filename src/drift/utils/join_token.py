"""Compact, human-shareable join tokens for a DRIFT-LLM private swarm.

A libp2p multiaddr like::

    /ip4/203.0.113.10/tcp/31337/p2p/12D3KooWAb3xY...

is precise but long and easy to mistype. A join token carries the same
information in a familiar URL shape::

    drift://12D3KooWAb3xY...@203.0.113.10:31337

The two forms round-trip: ``decode_join_token(encode_join_token(maddr))`` yields
an equivalent multiaddr. Only TCP addresses are represented (DRIFT-LLM speaks TCP);
the host type is inferred on decode -- an IPv4 literal maps back to ``/ip4``, a
bracketed ``[...]`` IPv6 literal to ``/ip6``, and anything else (a hostname) to
``/dns`` (which hivemind resolves to both A and AAAA records).
"""

import ipaddress
import re
from typing import Iterable, List

__all__ = [
    "SCHEME",
    "encode_join_token",
    "decode_join_token",
    "parse_join",
    "select_advertisable_maddrs",
]

SCHEME = "drift://"

# Host protocols we know how to shorten. dns4/dns6 collapse to /dns on decode,
# which resolves both address families and keeps the token free of family hints.
_HOST_PROTOCOLS = {"ip4", "ip6", "dns", "dns4", "dns6"}


def encode_join_token(maddr: str) -> str:
    """Convert a ``/ip4/.../tcp/<port>/p2p/<peer_id>`` multiaddr into a drift:// token."""
    parts = str(maddr).strip().split("/")
    # A well-formed maddr starts with "/", so split() yields a leading empty string.
    if len(parts) != 7 or parts[0] != "":
        raise ValueError(f"Cannot encode multiaddr {maddr!r}: expected /<host_proto>/<host>/tcp/<port>/p2p/<peer_id>")

    host_proto, host, transport, port, p2p, peer_id = parts[1:]
    if host_proto not in _HOST_PROTOCOLS:
        raise ValueError(f"Cannot encode multiaddr {maddr!r}: unsupported host protocol /{host_proto}")
    if transport != "tcp":
        raise ValueError(f"Cannot encode multiaddr {maddr!r}: only /tcp addresses are supported, got /{transport}")
    if p2p != "p2p":
        raise ValueError(f"Cannot encode multiaddr {maddr!r}: expected a /p2p/<peer_id> suffix")
    if not port.isdigit():
        raise ValueError(f"Cannot encode multiaddr {maddr!r}: port {port!r} is not numeric")

    host_part = f"[{host}]" if host_proto == "ip6" else host
    return f"{SCHEME}{peer_id}@{host_part}:{port}"


def decode_join_token(token: str) -> str:
    """Convert a drift:// token back into an equivalent libp2p multiaddr."""
    token = str(token).strip()
    if not token.lower().startswith(SCHEME):
        raise ValueError(f"Not a join token (missing {SCHEME!r} scheme): {token!r}")
    body = token[len(SCHEME) :]

    peer_id, sep, address = body.partition("@")
    if not sep or not peer_id or not address:
        raise ValueError(f"Malformed join token {token!r}: expected {SCHEME}<peer_id>@<host>:<port>")

    if address.startswith("["):  # bracketed IPv6 literal, e.g. [::1]:31337
        host_close = address.find("]")
        if host_close == -1 or not address[host_close + 1 :].startswith(":"):
            raise ValueError(f"Malformed join token {token!r}: unterminated IPv6 host")
        host = address[1:host_close]
        port = address[host_close + 2 :]
        host_proto = "ip6"
    else:
        host, sep, port = address.rpartition(":")
        if not sep:
            raise ValueError(f"Malformed join token {token!r}: expected <host>:<port>")
        host_proto = _infer_host_protocol(host)

    if not port.isdigit():
        raise ValueError(f"Malformed join token {token!r}: port {port!r} is not numeric")
    return f"/{host_proto}/{host}/tcp/{port}/p2p/{peer_id}"


def parse_join(value: str) -> List[str]:
    """Turn a --join value into a list of multiaddrs.

    Accepts drift:// tokens and raw multiaddrs, singly or comma-separated, so a
    user can paste whichever form they have.
    """
    maddrs = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if item.lower().startswith(SCHEME):
            maddrs.append(decode_join_token(item))
        elif item.startswith("/"):
            maddrs.append(item)  # already a multiaddr
        else:
            raise ValueError(f"Cannot parse join address {item!r}: expected a {SCHEME} token or a /... multiaddr")
    if not maddrs:
        raise ValueError(f"No join addresses found in {value!r}")
    return maddrs


def _infer_host_protocol(host: str) -> str:
    try:
        version = ipaddress.ip_address(host).version
    except ValueError:
        return "dns"
    return "ip6" if version == 6 else "ip4"


# libp2p peer ids: base58btc (Qm... CIDv0) or the newer base32/base58 forms. We
# only need to reject obviously-relay/loopback maddrs, so keep this permissive.
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
_MADDR_HOST_RE = re.compile(r"^/(ip4|ip6|dns|dns4|dns6)/([^/]+)/tcp/(\d+)/p2p/([^/]+)$")


def select_advertisable_maddrs(maddrs: Iterable[str], *, include_loopback: bool = False) -> List[str]:
    """Order visible multiaddrs best-first for advertising as a join address.

    Drops relay/circuit and (by default) loopback addresses, then prefers global
    IPv4 > private/LAN IPv4 > IPv6 > hostnames so the shared token points at the
    most broadly reachable interface. Returns only plain /tcp/.../p2p addresses.
    """

    def rank(maddr: str) -> int:
        match = _MADDR_HOST_RE.match(maddr)
        proto, host = match.group(1), match.group(2)
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            return 4  # hostname: reachable but least explicit
        if ip.version == 4:
            return 1 if ip.is_global else 2
        return 3  # IPv6 literal

    candidates = []
    for maddr in maddrs:
        maddr = str(maddr)
        match = _MADDR_HOST_RE.match(maddr)
        if match is None:  # relay/circuit, quic, ws, or otherwise non-plain-tcp
            continue
        host = match.group(2)
        if not include_loopback and host in _LOOPBACK_HOSTS:
            continue
        try:
            if not include_loopback and ipaddress.ip_address(host).is_loopback:
                continue
        except ValueError:
            pass  # hostname; keep it
        candidates.append(maddr)

    return sorted(candidates, key=rank)
