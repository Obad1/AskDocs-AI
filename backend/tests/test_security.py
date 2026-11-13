"""Regression tests for local-peer and data-root security boundaries."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from app.core.security import LocalOnlyAuth, ensure_within_root, safe_join


class LocalOnlyAuthTests(unittest.IsolatedAsyncioTestCase):
    async def test_accepts_direct_loopback_peers(self):
        auth = LocalOnlyAuth()
        for host in ("127.0.0.1", "::1"):
            request = SimpleNamespace(client=SimpleNamespace(host=host), headers={})
            self.assertEqual(await auth(request), "local")

    async def test_rejects_remote_peer_even_with_spoofed_forwarded_header(self):
        auth = LocalOnlyAuth()
        request = SimpleNamespace(
            client=SimpleNamespace(host="203.0.113.10"),
            headers={"x-forwarded-for": "127.0.0.1"},
        )
        with self.assertRaises(HTTPException) as raised:
            await auth(request)
        self.assertEqual(raised.exception.status_code, 403)

    async def test_rejects_forwarded_headers_and_missing_peer(self):
        auth = LocalOnlyAuth()
        requests = (
            SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"), headers={"forwarded": "for=127.0.0.1"}),
            SimpleNamespace(client=None, headers={}),
        )
        for request in requests:
            with self.subTest(request=request), self.assertRaises(HTTPException):
                await auth(request)

    async def test_optional_auth_mode_remains_explicitly_disabled(self):
        request = SimpleNamespace(client=SimpleNamespace(host="203.0.113.10"), headers={})
        self.assertEqual(await LocalOnlyAuth(require_local=False)(request), "local")


class DataRootTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "data"
        self.root.mkdir()
        self.settings = SimpleNamespace(data_root=str(self.root))
        self.settings_patch = patch("app.core.security.get_settings", return_value=self.settings)
        self.settings_patch.start()
        self.addCleanup(self.settings_patch.stop)
        self.addCleanup(self.temp.cleanup)

    def test_safe_join_accepts_child_and_rejects_escape_components(self):
        self.assertEqual(safe_join(self.root, "workspaces", "item.json"), self.root / "workspaces" / "item.json")
        for component in ("..", "../outside", "/tmp/outside", "C:\\outside", "nested\\file", "bad\x00name"):
            with self.subTest(component=component), self.assertRaises(HTTPException):
                safe_join(self.root, component)

    def test_safe_join_rejects_base_outside_configured_root(self):
        with self.assertRaises(HTTPException):
            safe_join(self.root.parent, "outside")

    def test_safe_join_resolves_symlink_escape(self):
        outside = Path(self.temp.name) / "outside"
        outside.mkdir()
        link = self.root / "link"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks are unavailable")
        with self.assertRaises(HTTPException):
            safe_join(self.root, "link", "file.txt")

    def test_ensure_within_root_resolves_paths_and_rejects_escape(self):
        child = self.root / "child.txt"
        self.assertEqual(ensure_within_root(child), child)
        with self.assertRaises(HTTPException):
            ensure_within_root(self.root.parent / "outside.txt")


if __name__ == "__main__":
    unittest.main()
