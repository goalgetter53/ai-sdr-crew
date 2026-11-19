from __future__ import annotations

import ipaddress

import httpx
import pytest
from hypothesis import given
from hypothesis import strategies as st

from ai_sdr_crew.tools.fetch import FetchBlocked, FetchFailed, SafeFetcher, is_public_ip
from ai_sdr_crew.tools.web_fixture import FAKE_PUBLIC_IP, FixtureWeb

PUBLIC = "93.184.215.14"


def fetcher_for(handler, resolve: dict[str, list[str]] | None = None, **kw) -> SafeFetcher:
    table = resolve or {}

    def resolver(host: str) -> list[str]:
        if host in table:
            return table[host]
        try:
            ipaddress.ip_address(host)
            return [host]
        except ValueError:
            return [PUBLIC]

    return SafeFetcher(resolver=resolver, transport=httpx.MockTransport(handler), **kw)


def ok_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, text="<p>hello</p>", headers={"content-type": "text/html"})


@pytest.mark.parametrize(
    "addr",
    [
        "127.0.0.1", "10.1.2.3", "172.16.0.9", "192.168.1.1", "169.254.169.254", "0.0.0.0",
        "100.64.0.1", "::1", "fc00::1", "fe80::1", "::ffff:127.0.0.1", "224.0.0.1",
        "2002:7f00:0001::1",
    ],
)  # fmt: skip
def test_non_public_addresses_rejected(addr: str) -> None:
    assert not is_public_ip(addr)


def test_public_address_allowed() -> None:
    assert is_public_ip(PUBLIC)
    assert is_public_ip("2606:4700:4700::1111")


@given(st.ip_addresses(v=4, network="10.0.0.0/8") | st.ip_addresses(v=4, network="192.168.0.0/16"))
def test_private_ipv4_property(ip: ipaddress.IPv4Address) -> None:
    assert not is_public_ip(str(ip))


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "gopher://example.com/",
        "ftp://example.com/x",
        "http://user:pw@site.example/",
        "http://localhost/",
        "http://metadata.google.internal/",
        "http://printer.local/",
        "http://site.example:8080/",
        "http://127.0.0.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
    ],
)
def test_policy_blocks_dangerous_urls(url: str) -> None:
    f = fetcher_for(ok_handler)
    with pytest.raises(FetchBlocked):
        f.fetch(url)


def test_hostname_resolving_to_private_ip_blocked() -> None:
    f = fetcher_for(ok_handler, {"rebind.example": ["10.0.0.5"]})
    with pytest.raises(FetchBlocked, match="non-public"):
        f.fetch("https://rebind.example/")


def test_any_private_address_in_dns_answer_blocks() -> None:
    f = fetcher_for(ok_handler, {"mixed.example": [PUBLIC, "127.0.0.1"]})
    with pytest.raises(FetchBlocked):
        f.fetch("https://mixed.example/")


def test_connection_pinned_to_validated_ip_with_host_header() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return ok_handler(request)

    f = fetcher_for(handler)
    result = f.fetch("https://www.acme.example/about?x=1")
    assert result.status == 200
    assert seen[0].url.host == PUBLIC
    assert seen[0].headers["host"] == "www.acme.example"
    assert seen[0].url.query == b"x=1"
    assert seen[0].extensions["sni_hostname"] == "www.acme.example"


def test_redirect_to_private_address_blocked() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/"})

    with pytest.raises(FetchBlocked):
        fetcher_for(handler).fetch("https://www.acme.example/")


def test_redirect_loop_capped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "/again"})

    with pytest.raises(FetchBlocked, match="too many redirects"):
        fetcher_for(handler, max_redirects=2).fetch("https://www.acme.example/")


def test_safe_redirect_followed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old":
            return httpx.Response(301, headers={"location": "/new"})
        return ok_handler(request)

    r = fetcher_for(handler).fetch("https://www.acme.example/old")
    assert r.final_url.endswith("/new")


def test_size_cap_truncates() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"a" * 5000, headers={"content-type": "text/plain"})

    r = fetcher_for(handler, max_bytes=1000).fetch("https://www.acme.example/")
    assert r.truncated and r.bytes_read == 1000


def test_content_type_allowlist() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"\x00\x01", headers={"content-type": "application/zip"})

    with pytest.raises(FetchBlocked, match="content type"):
        fetcher_for(handler).fetch("https://www.acme.example/")


def test_http_error_is_failure_not_block() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom", headers={"content-type": "text/html"})

    with pytest.raises(FetchFailed, match="HTTP 500"):
        fetcher_for(handler).fetch("https://www.acme.example/")


def test_guard_off_reaches_metadata_in_fixture_web() -> None:
    """Proves the ablation works: without the guard the fake metadata endpoint is reachable."""
    web = FixtureWeb({})
    f = SafeFetcher(resolver=web.resolver, transport=web.transport(), guard=False)
    assert "AKIA" in f.fetch("http://169.254.169.254/latest/").text
    assert web.resolver("www.any.example") == [FAKE_PUBLIC_IP]
