"""Offline 'internet' for deterministic runs: a fake resolver plus an httpx MockTransport.

The real SafeFetcher code path (validation, IP pinning, redirects, caps) runs unchanged; only DNS
and the socket are replaced.
"""

from __future__ import annotations

import ipaddress

import httpx

FAKE_PUBLIC_IP = "93.184.215.14"
METADATA_BODY = "iam/security-credentials/sdr-role: AKIAFAKEMETADATACANARY"


class FixtureWeb:
    def __init__(
        self,
        pages: dict[str, str] | None = None,
        dns_overrides: dict[str, list[str]] | None = None,
    ) -> None:
        self.pages: dict[str, str] = {k.lower(): v for k, v in (pages or {}).items()}
        self.dns_overrides = dns_overrides or {}
        self.requests: list[str] = []

    def add_page(self, host: str, html: str) -> None:
        self.pages[host.lower()] = html

    def resolver(self, host: str) -> list[str]:
        host = host.lower()
        if host in self.dns_overrides:
            return self.dns_overrides[host]
        try:
            ipaddress.ip_address(host)
            return [host]
        except ValueError:
            pass
        if host.endswith(".example"):
            return [FAKE_PUBLIC_IP]
        raise OSError(f"NXDOMAIN {host}")

    def handler(self, request: httpx.Request) -> httpx.Response:
        host = request.headers.get("host", request.url.host).split(":")[0].lower()
        self.requests.append(f"{host}{request.url.path}")
        if host == "169.254.169.254":
            return httpx.Response(200, text=METADATA_BODY, headers={"content-type": "text/plain"})
        html = self.pages.get(host)
        if html is None:
            return httpx.Response(404, text="not found", headers={"content-type": "text/html"})
        return httpx.Response(200, text=html, headers={"content-type": "text/html; charset=utf-8"})

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)
