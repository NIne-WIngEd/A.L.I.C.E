"""Physical package origins for exported evidence. No writes or normalization."""
from pathlib import Path

import contract as c

SOURCE_AUTHORITIES = ('tasks.json', 'draft.json', 'approval.json', 'approved.json',
                      'runtime_policy.json', 'source_binding_policy.json')


def package_projections(package, authorities=SOURCE_AUTHORITIES):
    package = Path(package)
    paths = {'package-manifest.json': package / 'PACKAGE_MANIFEST.json'}
    for name in authorities:
        c.safe_relative(name)
        c.require('/' not in name, 'Authority must be a package-owned basename')
        paths['authority/' + name] = package / 'authority' / name
    return paths


def source_package(rules):
    c.full_digest(rules['source_workload_sha256'])
    name = rules['source_package_name']
    c.safe_relative(name)
    c.require('/' not in name, 'Source package name')
    return c.ROOT / 'packages' / c.CALIBRATION_ID / rules['source_workload_sha256'] / name


def source_paths(run, rules):
    """Same package projection rule used by the production result collector."""
    projections = package_projections(source_package(rules))
    return {name: projections[name] if name in projections else Path(run) / c.safe_relative(name)
            for name in rules['source_critical_files']}


def verify_package_origin(package, expected_manifest_sha, expected_zip_sha):
    package = Path(package)
    manifest = package / 'PACKAGE_MANIFEST.json'
    c.require(manifest.is_file() and not manifest.is_symlink(), 'Original package manifest missing or linked')
    c.require(c.file_sha(manifest) == expected_manifest_sha, 'Original package manifest hash differs')
    archive = package.parent.with_suffix('.zip')
    c.require(archive.is_file() and not archive.is_symlink() and c.file_sha(archive) == expected_zip_sha,
              'Original package archive hash differs')
    c.verify_package(package)
    return {'package_manifest_sha256': expected_manifest_sha,
            'package_sha256': expected_zip_sha,
            'package_origin': package.as_posix()}
