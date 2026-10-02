
from pathlib import Path
import hashlib, json, os, platform, subprocess, sys, time

scratch = Path("/kaggle/working/mfm-nodata")
scratch.mkdir(mode=0o700, exist_ok=False)
helper_source = '"""Enter a same-UID, loopback-only network namespace before running bash.\n\nMagnolia\'s util-linux 2.23.2 has no --map-current-user. Mapping must happen\nin this process, before exec drops the new user namespace capabilities.\n"""\n\nfrom __future__ import annotations\n\nimport ctypes\nimport os\nimport socket\nimport sys\n\n\nCLONE_NEWUSER = 0x10000000\nCLONE_NEWNET = 0x40000000\n\n\ndef _write_once(path: str, value: str) -> None:\n    fd = os.open(path, os.O_WRONLY)\n    try:\n        data = value.encode("ascii")\n        if os.write(fd, data) != len(data):\n            raise OSError("short write to " + path)\n    finally:\n        os.close(fd)\n\n\ndef _enter() -> None:\n    uid, gid = os.geteuid(), os.getegid()\n    if uid == 0:\n        raise RuntimeError("refusing root caller before namespace creation")\n    if os.environ.get("MFM_HOST_UID") != str(uid) or os.environ.get("MFM_HOST_GID") != str(gid):\n        raise RuntimeError("outer UID/GID pins do not match caller")\n    host_netns = os.readlink("/proc/self/ns/net")\n    if os.environ.get("MFM_HOST_NETNS") != host_netns:\n        raise RuntimeError("outer network namespace pin does not match caller")\n\n    libc = ctypes.CDLL(None, use_errno=True)\n    libc.unshare.argtypes = [ctypes.c_int]\n    libc.unshare.restype = ctypes.c_int\n    if libc.unshare(CLONE_NEWUSER | CLONE_NEWNET) != 0:\n        err = ctypes.get_errno()\n        raise OSError(err, "unshare user+network namespace: " + os.strerror(err))\n\n    _write_once("/proc/self/uid_map", "{0} {0} 1\\n".format(uid))\n    _write_once("/proc/self/setgroups", "deny\\n")\n    _write_once("/proc/self/gid_map", "{0} {0} 1\\n".format(gid))\n    if os.geteuid() != uid or os.getegid() != gid:\n        raise RuntimeError("identity mapping did not preserve nonroot caller")\n    with open("/proc/self/uid_map", encoding="ascii") as uid_map:\n        observed_uid_map = uid_map.read().split()\n    with open("/proc/self/gid_map", encoding="ascii") as gid_map:\n        observed_gid_map = gid_map.read().split()\n    if observed_uid_map != [str(uid), str(uid), "1"]:\n        raise RuntimeError("UID map differs from the single identity mapping")\n    if observed_gid_map != [str(gid), str(gid), "1"]:\n        raise RuntimeError("GID map differs from the single identity mapping")\n    new_netns = os.readlink("/proc/self/ns/net")\n    interfaces = sorted(name for _, name in socket.if_nameindex())\n    if new_netns == host_netns or interfaces != ["lo"]:\n        raise RuntimeError("new namespace is not isolated to loopback")\n    print("namespace_uid={} namespace_gid={} namespace_netns={} interfaces=lo".format(\n        uid, gid, new_netns), flush=True)\n\n    # An inherited open socket would remain usable after a namespace change.\n    # Keep only Slurm\'s standard streams before starting the P2 check/work.\n    for entry in os.listdir("/proc/self/fd"):\n        if entry.isdigit() and int(entry) > 2:\n            try:\n                os.close(int(entry))\n            except OSError:\n                pass\n\n\ndef main() -> int:\n    if len(sys.argv) < 4 or sys.argv[1:3] != ["--", "/bin/bash"]:\n        print("usage: enter_private_network_namespace.py -- /bin/bash [args]", file=sys.stderr)\n        return 2\n    try:\n        _enter()\n        os.execv("/bin/bash", sys.argv[2:])\n    except (OSError, RuntimeError) as exc:\n        print("private namespace unavailable: " + str(exc), file=sys.stderr)\n        return 3\n    return 3\n\n\nif __name__ == "__main__":\n    sys.exit(main())\n'
assert hashlib.sha256(helper_source.encode()).hexdigest() == "1907d88ad39e5cb2f035f8812704d41972795f40a2195f7082a5d2305ad87c2d"
helper_path = scratch / "enter_namespace.py"
helper_path.write_text(helper_source)
inside = scratch / "inside.py"
inside.write_text("""
import json, os, socket, torch
assert sorted(n for _, n in socket.if_nameindex()) == ['lo']
assert os.readlink('/proc/self/ns/net') != os.environ['MFM_HOST_NETNS']
assert os.geteuid() != 0
print(json.dumps({'network_namespace_passed': True, 'uid': os.geteuid(),
    'gid': os.getegid(), 'torch_version': torch.__version__,
    'cuda_available_inside_isolation': torch.cuda.is_available(),
    'cuda_device_count': torch.cuda.device_count(),
    'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    'private_corpus_opened': False}), flush=True)
""")
uid = 1000 if os.geteuid() == 0 else os.geteuid()
gid = 1000 if os.geteuid() == 0 else os.getegid()
if os.geteuid() == 0:
    for path in (scratch, helper_path, inside):
        os.chown(path, uid, gid)
env = dict(os.environ, MFM_HOST_UID=str(uid), MFM_HOST_GID=str(gid),
    MFM_HOST_NETNS=os.readlink('/proc/self/ns/net'))
def drop():
    if os.geteuid() == 0:
        os.setgroups([])
        os.setgid(gid)
        os.setuid(uid)
start = time.monotonic()
attempt = subprocess.run([sys.executable, str(helper_path), '--', '/bin/bash', '-c',
    'exec "$1" "$2"', 'nodata-isolation', sys.executable, str(inside)],
    env=env, preexec_fn=drop, capture_output=True, text=True, timeout=120)
net_only = scratch / 'root_net_only.py'
net_only.write_text('''
import ctypes, os, socket, sys
assert os.geteuid() == 0
libc = ctypes.CDLL(None, use_errno=True)
libc.unshare.argtypes = [ctypes.c_int]
libc.unshare.restype = ctypes.c_int
if libc.unshare(0x40000000) != 0:
    err = ctypes.get_errno()
    raise OSError(err, 'root network namespace: ' + os.strerror(err))
assert sorted(n for _, n in socket.if_nameindex()) == ['lo']
assert os.readlink('/proc/self/ns/net') != os.environ['MFM_HOST_NETNS']
os.setgroups([])
os.setgid(1000)
os.setuid(1000)
assert os.geteuid() == 1000 and os.getegid() == 1000
for entry in os.listdir('/proc/self/fd'):
    if entry.isdigit() and int(entry) > 2:
        try:
            os.close(int(entry))
        except OSError:
            pass
os.execv(sys.executable, [sys.executable, sys.argv[1]])
''')
root_net = None
if os.geteuid() == 0:
    root_net = subprocess.run([sys.executable, str(net_only), str(inside)],
        env=env, capture_output=True, text=True, timeout=120)
# Public runtime metadata only. Even if isolation fails, this opens no dataset,
# checkpoint, source text, credentials or hosted model.
import torch
receipt = {'schema': 'mfm-kaggle-nodata-gpu-isolation-v1',
    'kaggle_owner': 'mkrayanyan', 'kernel': 'rayan-mfm-nodata-gpu-20261002',
    'requested_accelerator': 'NvidiaL4', 'requested_internet': False,
    'private_kernel_requested': True, 'datasets_attached': False,
    'private_corpus_opened': False, 'model_weights_opened': False,
    'actual_torch_version': torch.__version__, 'actual_cuda_available': torch.cuda.is_available(),
    'actual_cuda_device_count': torch.cuda.device_count(),
    'actual_devices': [{'name': torch.cuda.get_device_name(i),
        'capability': list(torch.cuda.get_device_capability(i)),
        'total_memory_bytes': torch.cuda.get_device_properties(i).total_memory}
        for i in range(torch.cuda.device_count())],
    'host_kernel': platform.release(), 'namespace_helper_sha256': '1907d88ad39e5cb2f035f8812704d41972795f40a2195f7082a5d2305ad87c2d',
    'namespace_exit_code': attempt.returncode,
    'namespace_stdout': attempt.stdout[-6000:], 'namespace_stderr': attempt.stderr[-2000:],
    'root_net_only_exit_code': None if root_net is None else root_net.returncode,
    'root_net_only_stdout': '' if root_net is None else root_net.stdout[-6000:],
    'root_net_only_stderr': '' if root_net is None else root_net.stderr[-2000:],
    'scope': 'two no-data isolation methods; no private fallback permitted',
    'elapsed_seconds': time.monotonic() - start, 'qualified_for_product': False}
canonical = json.dumps(receipt, sort_keys=True, separators=(',', ':')).encode()
receipt['record_sha256'] = hashlib.sha256(canonical).hexdigest()
output = Path('/kaggle/working/mfm-kaggle-nodata-receipt.json')
output.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, sort_keys=True), flush=True)
