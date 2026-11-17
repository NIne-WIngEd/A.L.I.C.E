"""Recheck the established owner boundary without repeating startup FD cleanup.

The original pinned _p2_guard MUST run before any protected path operation.
This verifier is only for an already-entered process that owns live file handles.
It checks the same namespace, mapping and effective identity against that entry
receipt; descriptor/path/hash custody remains the caller's separate obligation.
"""
from hashlib import sha256
from pathlib import Path
import traceback

GUARD_SHA256='f5822c3657b246a5b232f42bddf93b8026bda0b6a7e573835c1cd8c90136c1e4'


def verify_active_namespace(namespace, expected_isolation):
    if namespace.sys.platform!='linux' or sha256(Path(namespace.__file__).read_bytes()).hexdigest()!=GUARD_SHA256:
        raise ValueError('live namespace implementation differs')
    host_ns,uid,gid=namespace._namespace_identity()
    current=namespace.os.readlink('/proc/self/ns/net')
    interfaces=sorted(name for _,name in namespace.socket.if_nameindex())
    if current==host_ns or current!=namespace.os.environ.get('PERSONALITY_ISOLATED_NETNS') or interfaces!=['lo']:
        raise ValueError('live namespace or interface boundary differs')
    if namespace._read_map('/proc/self/uid_map')!=[str(uid),str(uid),'1'] or \
            namespace._read_map('/proc/self/gid_map')!=[str(gid),str(gid),'1']:
        raise ValueError('live namespace owner mapping differs')
    effective_uid,effective_gid=namespace.os.geteuid(),namespace.os.getegid()
    if effective_uid not in (0,uid) or effective_gid not in (0,gid):
        raise ValueError('live namespace effective owner differs')
    actual={'host_uid':uid,'host_gid':gid,'host_netns':host_ns,'isolated_netns':current,
            'interfaces':interfaces,'p2_effective_uid':effective_uid,'p2_effective_gid':effective_gid}
    if actual!=expected_isolation:
        raise ValueError('live namespace differs from original entry receipt')
    return actual


def public_exception_chain(error):
    """Retain causal frames/errno without messages, locals, source lines or full paths."""
    rows=[]
    seen=set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        rows.append({'exception_type':type(error).__name__,
                     'errno':error.errno if isinstance(error,OSError) and type(error.errno) is int else None,
                     'public_code_frames':[{'module':Path(frame.filename).name,'line':frame.lineno,'function':frame.name}
                                           for frame in traceback.extract_tb(error.__traceback__)]})
        error=error.__cause__ if error.__cause__ is not None else error.__context__
    return rows
