#!/bin/bash
# Transmitted as a literal file on SSH stdin. No PowerShell interpolation.
set -euo pipefail
umask 077
ACTION="$1"
PACKAGE_SHA="$2"
PYROOT="/modules/pkgs/common/python/3.11.5"
PY="$PYROOT/bin/python3.11"
PYLIB=""
for d in "$PYROOT/lib" "$PYROOT/lib64"; do
  if [ -e "$d/libpython3.11.so.1.0" ]; then PYLIB="$d"; break; fi
done
if [ -z "$PYLIB" ]; then echo 'Magnolia Python shared library missing' >&2; exit 127; fi
export LD_LIBRARY_PATH="$PYLIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PACKAGE_ROOT="/homes/01/mxrayan/rayan-compute/packages/alice-qwen38-a2-c926d9e355dd/$PACKAGE_SHA"
ARCHIVE="$PACKAGE_ROOT.zip"
"$PY" - "$ARCHIVE" "$PACKAGE_SHA" "$PACKAGE_ROOT" <<'PY'
import hashlib,json,os,sys,zipfile
from pathlib import Path,PurePosixPath
archive,expected,destination=Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3])
assert len(expected)==64 and all(c in '0123456789abcdef' for c in expected),'full package SHA required'
assert hashlib.sha256(archive.read_bytes()).hexdigest()==expected,'outer ZIP hash mismatch'
name='ALICE_MC10D_QWEN_ORIGIN_REVISION_v1.0.3'
with zipfile.ZipFile(archive) as z:
    names=z.namelist()
    assert len(names)==len(set(names)) and len(names)<100,'archive members'
    assert sum(x.file_size for x in z.infolist())<4*1024*1024,'archive size'
    for item in z.infolist():
        p=PurePosixPath(item.filename)
        assert not p.is_absolute() and p.parts[0]==name and '..' not in p.parts and '\\' not in item.filename,'archive path'
        assert (item.external_attr>>16)&0o170000 != 0o120000,'archive symlink'
    if not destination.exists():
        destination.mkdir(parents=True)
        z.extractall(destination)
root=destination/name
manifest=json.loads((root/'PACKAGE_MANIFEST.json').read_bytes())
listed=set()
for e in manifest['files']:
    p=root/e['path']
    assert p.resolve().is_relative_to(root.resolve()) and not p.is_symlink(),'manifest path'
    assert e['path'] not in listed,'manifest duplicate'
    listed.add(e['path'])
    assert p.stat().st_size==e['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==e['sha256'],'manifest hash'
observed={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
assert observed==listed|{'PACKAGE_MANIFEST.json'},'unmanifested package file'
PY
exec "$PY" "$PACKAGE_ROOT/ALICE_MC10D_QWEN_ORIGIN_REVISION_v1.0.3/remote_agent.py" "$ACTION" --package-sha "$PACKAGE_SHA"
