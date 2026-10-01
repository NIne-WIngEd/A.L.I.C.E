"""The legacy Magnolia namespace helper must fail before any private work."""

from __future__ import annotations

from contextlib import ExitStack
import io
import os
import unittest
from unittest.mock import patch

from scripts.mfm import enter_private_network_namespace as helper


class _Unshare:
    def __init__(self, result=0):
        self.result = result
        self.flags = []

    def __call__(self, flags):
        self.flags.append(flags)
        return self.result


class NamespaceHelperTests(unittest.TestCase):
    def test_identity_mapping_precedes_exec_and_requires_loopback(self):
        unshare = _Unshare()
        writes = []
        closed = []
        values = {
            "/proc/self/uid_map": "1905 1905 1\n",
            "/proc/self/gid_map": "100 100 1\n",
        }
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {
                "MFM_HOST_UID": "1905", "MFM_HOST_GID": "100",
                "MFM_HOST_NETNS": "net:[1]",
            }))
            stack.enter_context(patch.object(helper.os, "geteuid", return_value=1905))
            stack.enter_context(patch.object(helper.os, "getegid", return_value=100))
            stack.enter_context(patch.object(helper.os, "readlink", side_effect=["net:[1]", "net:[2]"]))
            stack.enter_context(patch.object(helper.os, "listdir", return_value=["0", "1", "2", "7"]))
            stack.enter_context(patch.object(helper.os, "close", side_effect=closed.append))
            stack.enter_context(patch.object(helper.socket, "if_nameindex", return_value=[(1, "lo")]))
            stack.enter_context(patch.object(helper.ctypes, "CDLL", return_value=type("Lib", (), {"unshare": unshare})()))
            stack.enter_context(patch.object(helper, "_write_once", side_effect=lambda p, v: writes.append((p, v))))
            stack.enter_context(patch("builtins.open", side_effect=lambda p, **_: io.StringIO(values[p])))
            helper._enter()
        self.assertEqual(unshare.flags, [helper.CLONE_NEWUSER | helper.CLONE_NEWNET])
        self.assertEqual(writes, [
            ("/proc/self/uid_map", "1905 1905 1\n"),
            ("/proc/self/setgroups", "deny\n"),
            ("/proc/self/gid_map", "100 100 1\n"),
        ])
        self.assertEqual(closed, [7])

    def test_root_caller_is_rejected_before_unshare(self):
        with patch.object(helper.os, "geteuid", return_value=0), \
             patch.object(helper.ctypes, "CDLL") as libc:
            with self.assertRaisesRegex(RuntimeError, "refusing root"):
                helper._enter()
        libc.assert_not_called()

    def test_failed_namespace_never_executes_bash(self):
        with patch.object(helper, "_enter", side_effect=OSError("denied")), \
             patch.object(helper.os, "execv") as execute, \
             patch.object(helper.sys, "argv", ["helper", "--", "/bin/bash", "-c", "private"]):
            self.assertEqual(helper.main(), 3)
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
