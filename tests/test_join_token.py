import pytest

from petals.utils.join_token import SCHEME, decode_join_token, encode_join_token, parse_join, select_advertisable_maddrs

PEER = "12D3KooWAb3xYcuspLm8sMbHhBc9x8pnwFy8Fq6iH1jK2example"


@pytest.mark.parametrize(
    "maddr, token",
    [
        (f"/ip4/203.0.113.10/tcp/31337/p2p/{PEER}", f"{SCHEME}{PEER}@203.0.113.10:31337"),
        (f"/ip6/2001:db8::1/tcp/31337/p2p/{PEER}", f"{SCHEME}{PEER}@[2001:db8::1]:31337"),
        (f"/dns/bootstrap.example.com/tcp/443/p2p/{PEER}", f"{SCHEME}{PEER}@bootstrap.example.com:443"),
    ],
)
def test_encode_matches_expected(maddr, token):
    assert encode_join_token(maddr) == token


@pytest.mark.parametrize(
    "maddr",
    [
        f"/ip4/203.0.113.10/tcp/31337/p2p/{PEER}",
        f"/ip6/2001:db8::1/tcp/31337/p2p/{PEER}",
        f"/ip6/::1/tcp/5/p2p/{PEER}",
        f"/dns/bootstrap.example.com/tcp/443/p2p/{PEER}",
    ],
)
def test_round_trip_maddr(maddr):
    assert decode_join_token(encode_join_token(maddr)) == maddr


def test_dns_family_variants_collapse_to_dns():
    # dns4/dns6 lose their family hint but still resolve correctly via /dns.
    token = encode_join_token(f"/dns4/host.example/tcp/80/p2p/{PEER}")
    assert decode_join_token(token) == f"/dns/host.example/tcp/80/p2p/{PEER}"


def test_ipv4_literal_infers_ip4():
    assert decode_join_token(f"{SCHEME}{PEER}@10.0.0.5:6000") == f"/ip4/10.0.0.5/tcp/6000/p2p/{PEER}"


@pytest.mark.parametrize(
    "bad",
    [
        "/ip4/203.0.113.10/udp/31337/p2p/" + PEER,  # non-tcp transport
        "/ip4/203.0.113.10/tcp/31337",  # missing /p2p
        "/foo/203.0.113.10/tcp/31337/p2p/" + PEER,  # unsupported host proto
        "/ip4/203.0.113.10/tcp/notaport/p2p/" + PEER,  # bad port
    ],
)
def test_encode_rejects_bad_maddrs(bad):
    with pytest.raises(ValueError):
        encode_join_token(bad)


@pytest.mark.parametrize(
    "bad",
    [
        "https://203.0.113.10:31337",  # wrong scheme
        f"{SCHEME}{PEER}",  # no @host
        f"{SCHEME}@203.0.113.10:31337",  # empty peer id
        f"{SCHEME}{PEER}@203.0.113.10",  # no port
        f"{SCHEME}{PEER}@203.0.113.10:xyz",  # bad port
        f"{SCHEME}{PEER}@[2001:db8::1]",  # unterminated / portless ipv6
    ],
)
def test_decode_rejects_bad_tokens(bad):
    with pytest.raises(ValueError):
        decode_join_token(bad)


def test_parse_join_accepts_tokens_and_maddrs_and_lists():
    raw = f"/ip4/203.0.113.10/tcp/31337/p2p/{PEER}"
    token = f"{SCHEME}{PEER}@198.51.100.7:31338"
    result = parse_join(f"{raw}, {token}")
    assert result == [raw, f"/ip4/198.51.100.7/tcp/31338/p2p/{PEER}"]


def test_parse_join_empty_raises():
    with pytest.raises(ValueError):
        parse_join("  ,  ")


def test_select_advertisable_prefers_global_ipv4_and_drops_loopback_and_relay():
    maddrs = [
        f"/ip4/127.0.0.1/tcp/31337/p2p/{PEER}",  # loopback -> dropped
        f"/ip6/2001:db8::1/tcp/31337/p2p/{PEER}",  # ipv6
        f"/ip4/192.168.1.20/tcp/31337/p2p/{PEER}",  # private LAN ipv4
        f"/ip4/8.8.8.8/tcp/31337/p2p/{PEER}",  # global ipv4 -> best
        f"/ip4/198.51.100.9/tcp/31337/p2p/{PEER}/p2p-circuit/p2p/{PEER}",  # relay -> dropped
    ]
    ranked = select_advertisable_maddrs(maddrs)
    assert ranked[0] == f"/ip4/8.8.8.8/tcp/31337/p2p/{PEER}"
    assert f"/ip4/192.168.1.20/tcp/31337/p2p/{PEER}" in ranked
    assert all("127.0.0.1" not in m and "p2p-circuit" not in m for m in ranked)


def test_select_advertisable_can_keep_loopback_for_local_swarm():
    maddrs = [f"/ip4/127.0.0.1/tcp/31337/p2p/{PEER}"]
    assert select_advertisable_maddrs(maddrs, include_loopback=True) == maddrs
    assert select_advertisable_maddrs(maddrs) == []
