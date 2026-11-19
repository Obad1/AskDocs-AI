"""Regression: forwarded headers must not bypass LocalOnlyAuth."""
import asyncio
import unittest

from fastapi import HTTPException
from starlette.requests import Request

from app.core.security import LocalOnlyAuth


def request_for(host, headers=()):
    return Request({
        "type": "http",
        "method": "GET",
        "path": "/api/v1/ingest",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers],
        "client": (host, 8443) if host is not None else None,
        "server": ("localhost", 8000),
        "scheme": "http",
        "query_string": b"",
        "http_version": "1.1",
    })


class LocalOnlyTests(unittest.TestCase):
    def test_loopback_allowed(self):
        for host in ("127.0.0.1", "::1", "localhost", None):
            with self.subTest(host=host):
                self.assertEqual(asyncio.run(LocalOnlyAuth()(request_for(host))), "local")

    def test_remote_denied_even_with_spoofed_forwarded_headers(self):
        for headers in ((), (("x-forwarded-for", "127.0.0.1"),), (("origin", "http://localhost"),)):
            with self.subTest(headers=headers):
                with self.assertRaises(HTTPException) as caught:
                    asyncio.run(LocalOnlyAuth()(request_for("203.0.113.42", headers)))
                self.assertEqual(caught.exception.status_code, 403)

    def test_disabled_local_requirement_is_explicit(self):
        self.assertEqual(
            asyncio.run(LocalOnlyAuth(require_local=False)(request_for("203.0.113.42"))),
            "local",
        )


if __name__ == "__main__":
    unittest.main()
