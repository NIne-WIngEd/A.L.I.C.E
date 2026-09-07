"""Read existing Magnolia runtime metadata and available container tools; no execution recovery."""
import argparse
import base64
import json
import os
from pathlib import Path
import sys
import uuid
import zipfile

import collect_transport as transport

ROOT = Path('/homes/01/mxrayan/rayan-compute')
RUN_ID = 'alice-qwen38-a3-750881d854fa'


def policy():
    return REMOTE_POLICY if 'REMOTE_POLICY' in globals() else json.loads((Path(__file__).parent / 'inspection-policy.json').read_text())


def state_inventory(vault):
    return {a: transport.file_record(Path(vault) / ('tools/alice-astra/qwen-fallback-' + a + '/controller-state.json')) for a in ('a1', 'a2', 'a3')}


def source_inventory():
    run = ROOT / 'runs' / RUN_ID
    return {name: transport.file_record(run / name) for name in policy()['source_files']}


def verify_runtime_files():
    run = ROOT / 'runs' / RUN_ID
    paths = []
    for item in policy()['runtime_elf_files']:
        path = run / 'runtime' / item['path']
        # Runtime archive aliases may be symlinks; only targets inside this exact runtime are eligible.
        if not path.is_file() or not path.resolve().is_relative_to((run / 'runtime').resolve()): raise ValueError('Runtime path absent or escaped: ' + item['path'])
        raw = path.read_bytes()
        if len(raw) != item['bytes'] or transport.sha(raw) != item['sha256']: raise ValueError('Runtime bytes changed: ' + item['path'])
        if raw[:4] != b'\x7fELF': raise ValueError('Expected ELF object: ' + item['path'])
        paths.append(path.as_posix())
    return paths


def remote():
    report = {'schema': 'alice.magnolia.runtime-route.remote.v1', 'started_at': transport.now(),
        'observation_scope': 'LOGIN_NODE_ONLY; available routes require separate startup qualification before any model job',
        'source_result_zip_sha256': policy()['source_result_zip_sha256'], 'source_run_id': RUN_ID,
        'before': source_inventory(), 'commands': [], 'model_jobs_submitted': 0,
        'model_downloads': 0, 'model_services_started': 0, 'inference_requests': 0,
        'package_staged_on_host': False, 'runtime_or_system_files_modified': False}
    try:
        for name, expected in policy()['source_files'].items():
            if report['before'][name].get('sha256') != expected: raise ValueError('Closed source evidence changed: ' + name)
        paths = verify_runtime_files()
        report['runtime_elf_hashes_verified'] = len(paths)
        commands = [(['uname', '-srmo'], 10), (['getconf', 'GNU_LIBC_VERSION'], 10),
            (['readelf', '--wide', '--program-headers', '--dynamic', '--version-info', *paths], 20),
            (['bash', '-lc', 'type -a apptainer singularity podman bwrap proot'], 15),
            (['bash', '-lc', 'if type module; then module -t avail; else exit 127; fi'], 20),
            (['bash', '-lc', 'for tool in apptainer singularity podman bwrap proot; do if command -v "$tool"; then "$tool" --version; fi; done'], 20)]
        for argv, limit in commands:
            receipt = transport.capture(argv, limit)
            report['commands'].append(receipt)
            print(json.dumps({'event': 'command', 'receipt': receipt}), flush=True)
        report['status'] = 'COLLECTED'
        report['command_failures'] = sum(x['exit_code'] != 0 for x in report['commands'])
    except Exception as exc:
        report.update(status='SOURCE_OR_INSPECTION_STOP', error_class=type(exc).__name__, message=str(exc))
    report['after'] = source_inventory()
    report['selected_source_files_unchanged'] = report['before'] == report['after']
    report['finished_at'] = transport.now()
    print(json.dumps({'event': 'report', 'report': report}), flush=True)


def remote_script():
    # Reuse the checked stdin bootstrap and bounded raw-byte transport implementation.
    old = transport.remote_script().decode()
    prefix = old.split("exec \"$PY\" -B - <<'ALICE_TELEMETRY_TRANSPORT_PY'\n", 1)[0]
    dependency = Path(transport.__file__).read_text()
    own = Path(__file__).read_text().rsplit("\nif __name__ == '__main__':", 1)[0]
    body = 'import types,sys\nm=types.ModuleType("collect_transport")\nexec(' + repr(dependency) + ',m.__dict__)\nsys.modules["collect_transport"]=m\n'
    body += 'scope={"__name__":"runtime_route_inspection","REMOTE_POLICY":' + repr(policy()) + '}\n'
    body += 'exec(' + repr(own) + ',scope)\nscope["remote"]()\n'
    return (prefix + "exec \"$PY\" -B - <<'ALICE_RUNTIME_ROUTE_PY'\n" + body + 'ALICE_RUNTIME_ROUTE_PY\n').encode()


def collect(output_root, vault, command=transport.run_bounded):
    out = Path(output_root) / ('ALICE_RuntimeRoute_' + uuid.uuid4().hex[:12]); out.mkdir(parents=True)
    before = state_inventory(vault); script = remote_script()
    (out / 'remote-input.sh').write_bytes(script)
    receipt = transport.capture(transport.ssh_argv(Path.home() / '.ssh/rayan_magnolia_ed25519'), 150, command=command, input=script)
    for name in ('stdout', 'stderr'): (out / ('ssh.' + name + '.bin')).write_bytes(base64.b64decode(receipt[name]['base64']))
    (out / 'transport.json').write_text(json.dumps(receipt, indent=2) + '\n')
    report = None; errors = []
    for i, line in enumerate(base64.b64decode(receipt['stdout']['base64']).splitlines()):
        try:
            event = json.loads(line)
            if event.get('event') == 'report': report = event['report']
        except (ValueError, AttributeError) as exc: errors.append({'line': i, 'error': str(exc)})
    after = state_inventory(vault)
    result = {'schema': 'alice.magnolia.runtime-route.result.v1', 'report': report,
        'status': report['status'] if receipt['exit_code'] == 0 and report is not None else 'INCOMPLETE_TRANSPORT',
        'local_states_before': before, 'local_states_after': after, 'local_states_unchanged': before == after,
        'remote_script_sha256': transport.sha(script), 'parse_errors': errors,
        'execution_eligibility_inferred': False, 'scientific_outcome': 'NOT_EVALUATED'}
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    files = [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': transport.sha(p.read_bytes())} for p in sorted(out.iterdir())]
    (out / 'EVIDENCE_MANIFEST.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
    archive = out.parent / (out.name + '.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()): z.write(p, p.name)
    print('TRACE_ZIP=' + str(archive), flush=True)
    print('TRACE_SHA256=' + transport.sha(archive.read_bytes()), flush=True)
    print('inspection_status=' + result['status'], flush=True)
    if report: print('provider_command_failures=' + str(report.get('command_failures', 'unknown')), flush=True)
    return result, archive


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', type=Path, default=Path.home() / 'Downloads')
    p.add_argument('--vault-root', type=Path, default=Path(r'C:\ALICE_Vault'))
    args = p.parse_args()
    print('action=READ_ONLY_EXISTING_RUNTIME_AND_ROUTE_SURVEY; six bounded commands; no allocation, download, service or inference', flush=True)
    result, _ = collect(args.output_root, args.vault_root)
    print('Return TRACE_ZIP even when tools are unavailable. Keep a3 closed; no reset or new execution.', flush=True)
    return 0 if result['status'] == 'COLLECTED' and result['local_states_unchanged'] else 74


if __name__ == '__main__': sys.exit(main())
