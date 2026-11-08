"""Ongoing isolation verification must preserve the process's owned descriptors."""
from contextlib import ExitStack
from hashlib import sha256
import importlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.alice_personality.n1.manifest_structure_audit import _PinnedFile

try:
    api=importlib.import_module('scripts.eipm.gemma_n0.live_namespace_custody')
except ModuleNotFoundError as error:
    if error.name!='scripts.eipm.gemma_n0.live_namespace_custody':raise
    api=None


def fixture():
    root=Path(__file__).resolve().parents[2]
    state={'host_uid':1905,'host_gid':100,'host_netns':'net:[10]',
           'isolated_netns':'net:[11]','interfaces':['lo'],'p2_effective_uid':0,'p2_effective_gid':0}
    fake_os=SimpleNamespace(environ={'PERSONALITY_ISOLATED_NETNS':'net:[11]'},
        readlink=lambda _: 'net:[11]',geteuid=lambda:0,getegid=lambda:0)
    namespace=SimpleNamespace(__file__=str(root/'scripts/eipm/n1/enter_private_network_namespace.py'),
        sys=SimpleNamespace(platform='linux'),os=fake_os,
        socket=SimpleNamespace(if_nameindex=lambda:[(1,'lo')]),
        _namespace_identity=lambda:('net:[10]',1905,100),
        _read_map=lambda p:['1905','1905','1'] if p.endswith('uid_map') else ['100','100','1'])
    def destructive_entry():raise AssertionError('startup cleanup cannot run during live custody')
    namespace._p2_guard=destructive_entry
    namespace._close_inherited_descriptors=destructive_entry
    return namespace,state


def test_live_check_retains_actual_pinned_descriptor_and_exit_validation(tmp_path):
    assert api is not None, 'Distinct non-destructive live namespace verifier is missing'
    path=(tmp_path/'public.json').resolve()
    path.write_bytes(b'{"public":true}\n')
    namespace,state=fixture()
    with ExitStack() as stack:
        pinned=stack.enter_context(_PinnedFile(path,sha256(path.read_bytes()).hexdigest(),1024))
        for _ in range(4):
            assert api.verify_active_namespace(namespace,state)==state
            os.fstat(pinned.stream.fileno())
            pinned.verify()
        assert not pinned.stream.closed


@pytest.mark.parametrize('change',['host','namespace','interfaces','mapping','effective','expected','implementation','platform'])
def test_isolation_drift_refuses_without_closing_owned_files(tmp_path,change):
    assert api is not None
    namespace,state=fixture()
    if change=='host':namespace._namespace_identity=lambda:('net:[11]',1905,100)
    elif change=='namespace':namespace.os.environ['PERSONALITY_ISOLATED_NETNS']='net:[12]'
    elif change=='interfaces':namespace.socket.if_nameindex=lambda:[(1,'lo'),(2,'eth0')]
    elif change=='mapping':namespace._read_map=lambda _:['0','0','1']
    elif change=='effective':namespace.os.geteuid=lambda:999
    elif change=='expected':state['isolated_netns']='net:[12]'
    elif change=='implementation':namespace.__file__=__file__
    else:namespace.sys.platform='win32'
    with (tmp_path/'owned').open('wb') as stream:
        with pytest.raises(ValueError):api.verify_active_namespace(namespace,state)
        stream.write(b'still owned')


def test_failure_chain_preserves_primary_and_cleanup_without_private_messages():
    assert api is not None
    try:
        try:raise ValueError('PRIVATE_TEACHING_CONTENT')
        finally:raise OSError(9,'PRIVATE_FILE_NAME')
    except BaseException as error:
        result=api.public_exception_chain(error)
    import json
    assert [row['exception_type'] for row in result]==['OSError','ValueError']
    assert result[0]['errno']==9
    assert 'PRIVATE_' not in json.dumps(result)
