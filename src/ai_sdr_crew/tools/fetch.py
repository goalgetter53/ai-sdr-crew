"""SSRF-safe URL fetcher.

Defences:
- scheme allowlist (http/https), no userinfo, port allowlist;
- every resolved address must be public (blocks loopback, RFC1918, link-local / cloud metadata,
  CGNAT, multicast, reserved, IPv4-mapped IPv6);
- the connection is pinned to the validated IP (Host header + SNI keep virtual hosting working),
  which closes the DNS-rebinding window between check and connect;
- redirects are followed manually and every hop is re-validated;
- byte cap, wall-clock deadline and a content-type allowlist.
"""

from __future__ import annotations

import ipaddress
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import SplitResult, urljoin, urlsplit, urlunsplit

import httpx

Resolver = Callable[[str], list[str]]

BLOCKED_HOST_SUFFIXES = (".localhost", ".local", ".internal", ".localdomain", ".home.arpa")
BLOCKED_HOSTS = frozenset({"localhost", "metadata", "metadata.google.internal"})
ALLOWED_CONTENT_TYPES = ("text/html", "text/plain", "application/xhtml+xml")


class FetchBlocked(Exception):
    """The request violates the fetch policy (SSRF guard, scheme, port, size, type)."""


class FetchFailed(Exception):
    """The request was allowed but did not succeed (DNS, HTTP error, timeout)."""


def system_resolver(host: str) -> list[str]:
    infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    return sorted({str(info[4][0]) for info in infos})


def is_public_ip(addr: str) -> bool:
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address = ipaddress.ip_address(addr.split("%")[0])
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        elif ip.sixtofour is not None:
            ip = ip.sixtofour
    return bool(
        ip.is_global
        and not ip.is_private
        and not ip.is_loopback
        and not ip.is_link_local
        and not ip.is_multicast
        and not ip.is_reserved
        and not ip.is_unspecified
    )


@dataclass(frozen=True)
class FetchResult:
    url: str
    final_url: str
    status: int
    content_type: str
    text: str
    truncated: bool
    bytes_read: int


class SafeFetcher:
    def __init__(
        self,
        *,
        resolver: Resolver = system_resolver,
        transport: httpx.BaseTransport | None = None,
        guard: bool = True,
        max_bytes: int = 512_000,
        timeout_s: float = 5.0,
        max_redirects: int = 3,
        allowed_ports: frozenset[int] = frozenset({80, 443}),
        user_agent: str = "ai-sdr-crew/0.1 (research bot)",
    ) -> None:
        self._resolver = resolver
        self._transport = transport
        self.guard = guard
        self.max_bytes = max_bytes
        self.timeout_s = timeout_s
        self.max_redirects = max_redirects
        self.allowed_ports = allowed_ports
        self.user_agent = user_agent

    def validate(self, url: str) -> tuple[SplitResult, str]:
        """Return the parsed URL and the validated IP to pin the connection to."""
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https"):
            raise FetchBlocked(f"scheme not allowed: {parts.scheme or '(none)'}")
        if parts.username or parts.password:
            raise FetchBlocked("credentials in URL are not allowed")
        host = (parts.hostname or "").rstrip(".").lower()
        if not host:
            raise FetchBlocked("missing host")
        if host in BLOCKED_HOSTS or host.endswith(BLOCKED_HOST_SUFFIXES):
            raise FetchBlocked(f"host not allowed: {host}")
        try:
            port = parts.port or (443 if parts.scheme == "https" else 80)
        except ValueError as e:
            raise FetchBlocked("invalid port") from e
        if port not in self.allowed_ports:
            raise FetchBlocked(f"port not allowed: {port}")
        try:
            addrs = self._resolver(host)
        except OSError as e:
            raise FetchFailed(f"DNS resolution failed for {host}") from e
        if not addrs:
            raise FetchFailed(f"no addresses for {host}")
        for addr in addrs:
            if not is_public_ip(addr):
                raise FetchBlocked(f"{host} resolves to non-public address {addr}")
        return parts, addrs[0]

    def _pinned(self, parts: SplitResult, ip: str) -> tuple[str, dict[str, str], dict[str, Any]]:
        host = (parts.hostname or "").rstrip(".").lower()
        ip_host = f"[{ip}]" if ":" in ip else ip
        netloc = f"{ip_host}:{parts.port}" if parts.port else ip_host
        url = urlunsplit((parts.scheme, netloc, parts.path or "/", parts.query, ""))
        host_header = f"{host}:{parts.port}" if parts.port else host
        extensions: dict[str, Any] = {"sni_hostname": host} if parts.scheme == "https" else {}
        return url, {"Host": host_header}, extensions

    def fetch(self, url: str) -> FetchResult:
        deadline = time.monotonic() + self.timeout_s
        current = url
        with httpx.Client(
            transport=self._transport,
            timeout=httpx.Timeout(self.timeout_s),
            follow_redirects=False,
            headers={"User-Agent": self.user_agent},
        ) as client:
            for _hop in range(self.max_redirects + 1):
                if self.guard:
                    parts, ip = self.validate(current)
                    req_url, headers, ext = self._pinned(parts, ip)
                else:
                    req_url, headers, ext = current, {}, {}
                request = client.build_request("GET", req_url, headers=headers, extensions=ext)
                try:
                    resp = client.send(request, stream=True)
                except httpx.HTTPError as e:
                    raise FetchFailed(f"request failed: {type(e).__name__}") from e
                try:
                    if resp.is_redirect:
                        location = resp.headers.get("location", "")
                        if not location:
                            raise FetchFailed("redirect without location")
                        current = urljoin(current, location)
                        continue
                    if resp.status_code >= 400:
                        raise FetchFailed(f"HTTP {resp.status_code}")
                    ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                    if self.guard and ctype not in ALLOWED_CONTENT_TYPES:
                        raise FetchBlocked(f"content type not allowed: {ctype or '(none)'}")
                    body, truncated = self._read_capped(resp, deadline)
                    encoding = resp.encoding or "utf-8"
                    return FetchResult(
                        url=url,
                        final_url=current,
                        status=resp.status_code,
                        content_type=ctype,
                        text=body.decode(encoding, errors="replace"),
                        truncated=truncated,
                        bytes_read=len(body),
                    )
                finally:
                    resp.close()
        raise FetchBlocked(f"too many redirects (>{self.max_redirects})")

    def _read_capped(self, resp: httpx.Response, deadline: float) -> tuple[bytes, bool]:
        chunks: list[bytes] = []
        total = 0
        for chunk in resp.iter_bytes():
            if time.monotonic() > deadline:
                raise FetchFailed("timeout while reading body")
            if total + len(chunk) > self.max_bytes:
                chunks.append(chunk[: self.max_bytes - total])
                return b"".join(chunks), True
            chunks.append(chunk)
            total += len(chunk)
        return b"".join(chunks), False
