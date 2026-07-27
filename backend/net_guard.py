"""Outbound fetch guard.

VaultMind fetches URLs from several places that are not fully under the
user's control: pasted links, DuckDuckGo results, links discovered inside
scraped pages, and profile URLs. Any of those can point back at the
machine VaultMind runs on, or at another host on the private network.

Without a check, /ingest-url will fetch http://127.0.0.1:… or a cloud
metadata endpoint and store the response in the vault, where it can be
read back out through the normal query path.
"""

import ipaddress
import socket
from urllib.parse import urlparse


class BlockedURL(Exception):
    """Raised when a URL points somewhere the app must not fetch."""


def assert_fetchable_url(url: str) -> str:
    """Return `url` if it is safe to fetch, else raise BlockedURL.

    Known limitation: this resolves the hostname and checks the resulting
    addresses, but a hostile DNS server could answer with a public address
    here and a private one for the connection that follows. Closing that
    means pinning the resolved address through the socket, which requests
    does not expose.
    """
    parsed = urlparse((url or "").strip())

    if parsed.scheme not in ("http", "https"):
        raise BlockedURL(
            f"Only http and https URLs can be fetched "
            f"(got {parsed.scheme or 'no scheme'})."
        )
    if not parsed.hostname:
        raise BlockedURL("URL has no host.")

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        addrinfo = socket.getaddrinfo(parsed.hostname, port)
    except socket.gaierror as e:
        raise BlockedURL(f"Could not resolve {parsed.hostname}: {e}")

    for _, _, _, _, sockaddr in addrinfo:
        ip = ipaddress.ip_address(sockaddr[0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            raise BlockedURL(
                f"{parsed.hostname} resolves to {ip}, which is on the local "
                f"machine or a private network."
            )
    return url


def is_fetchable(url: str) -> bool:
    """Boolean form of assert_fetchable_url, for filtering lists of links."""
    try:
        assert_fetchable_url(url)
        return True
    except BlockedURL:
        return False
