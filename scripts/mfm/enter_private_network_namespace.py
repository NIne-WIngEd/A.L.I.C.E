"""Enter a same-UID, loopback-only network namespace before running bash.

Magnolia's util-linux 2.23.2 has no --map-current-user. Mapping must happen
in this process, before exec drops the new user namespace capabilities.
"""

from __future__ import annotations

import ctypes
import os
import socket
import sys


CLONE_NEWUSER = 0x10000000
CLONE_NEWNET = 0x40000000


def _write_once(path: str, value: str) -> None:
    fd = os.open(path, os.O_WRONLY)
    try:
        data = value.encode("ascii")
        if os.write(fd, data) != len(data):
            raise OSError("short write to " + path)
    finally:
        os.close(fd)


def _enter() -> None:
    uid, gid = os.geteuid(), os.getegid()
    if uid == 0:
        raise RuntimeError("refusing root caller before namespace creation")
    if os.environ.get("MFM_HOST_UID") != str(uid) or os.environ.get("MFM_HOST_GID") != str(gid):
        raise RuntimeError("outer UID/GID pins do not match caller")
    host_netns = os.readlink("/proc/self/ns/net")
    if os.environ.get("MFM_HOST_NETNS") != host_netns:
        raise RuntimeError("outer network namespace pin does not match caller")

    libc = ctypes.CDLL(None, use_errno=True)
    libc.unshare.argtypes = [ctypes.c_int]
    libc.unshare.restype = ctypes.c_int
    if libc.unshare(CLONE_NEWUSER | CLONE_NEWNET) != 0:
        err = ctypes.get_errno()
        raise OSError(err, "unshare user+network namespace: " + os.strerror(err))

    _write_once("/proc/self/uid_map", "{0} {0} 1\n".format(uid))
    _write_once("/proc/self/setgroups", "deny\n")
    _write_once("/proc/self/gid_map", "{0} {0} 1\n".format(gid))
    if os.geteuid() != uid or os.getegid() != gid:
        raise RuntimeError("identity mapping did not preserve nonroot caller")
    with open("/proc/self/uid_map", encoding="ascii") as uid_map:
        observed_uid_map = uid_map.read().split()
    with open("/proc/self/gid_map", encoding="ascii") as gid_map:
        observed_gid_map = gid_map.read().split()
    if observed_uid_map != [str(uid), str(uid), "1"]:
        raise RuntimeError("UID map differs from the single identity mapping")
    if observed_gid_map != [str(gid), str(gid), "1"]:
        raise RuntimeError("GID map differs from the single identity mapping")
    new_netns = os.readlink("/proc/self/ns/net")
    interfaces = sorted(name for _, name in socket.if_nameindex())
    if new_netns == host_netns or interfaces != ["lo"]:
        raise RuntimeError("new namespace is not isolated to loopback")
    print("namespace_uid={} namespace_gid={} namespace_netns={} interfaces=lo".format(
        uid, gid, new_netns), flush=True)

    # An inherited open socket would remain usable after a namespace change.
    # Keep only Slurm's standard streams before starting the P2 check/work.
    for entry in os.listdir("/proc/self/fd"):
        if entry.isdigit() and int(entry) > 2:
            try:
                os.close(int(entry))
            except OSError:
                pass


def main() -> int:
    if len(sys.argv) < 4 or sys.argv[1:3] != ["--", "/bin/bash"]:
        print("usage: enter_private_network_namespace.py -- /bin/bash [args]", file=sys.stderr)
        return 2
    try:
        _enter()
        os.execv("/bin/bash", sys.argv[2:])
    except (OSError, RuntimeError) as exc:
        print("private namespace unavailable: " + str(exc), file=sys.stderr)
        return 3
    return 3


if __name__ == "__main__":
    sys.exit(main())
