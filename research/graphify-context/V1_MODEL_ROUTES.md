# V1 model source routes

The existing Graphify code graph was extracted from an N0 branch. It is not a
current code graph for the Gemma source derivative, MFM specialist, or FBM
replay seed. `V1_MODEL_ROUTES.json` is a small, branch-qualified pointer index
for those three V1 surfaces. It is not a model assessment or a new authority.

From the Graphify research worktree, check the three **live** source heads
before using a pointer:

```bash
python scripts/context/build_v1_model_routes.py --check
python scripts/context/build_v1_model_routes.py --query 'mfm training'
```

A query result contains the exact source commit, path, and Git blob. Read the
original source before acting on a consequential claim:

```bash
git show '<commit>:<path>'
```

If the check reports `STALE_LOCAL_REF`, fetch the named source branch. If it
reports `STALE_V1_ROUTES`, regenerate and commit the index:

```bash
python scripts/context/build_v1_model_routes.py --write
```

The index deliberately excludes model weights, private material, generated
training cases, and inferred relations. This cheap route avoids importing the
Graphify package into the three model branches. A V1 graph would need its own
exact-source extraction receipts and freshness checks before use.

## Current receipt boundary

The foundation's `alice-gemma4-v1-source-v1` receipt differs from the older
MFM baseline runner's `mfm-gemma4-pretrained-source-v1` receipt. The older
`run_gemma4_base_behavior.py` checks the latter schema and its exact fields;
do not hand it a foundation receipt. Create and verify its own receipt or
adapt that runner with a reviewed, tested contract before baseline execution.
The source pointers above lead to both validators.

The owner reported all eight pinned SHA-256 checks as `OK` on Magnolia on
2026-09-30. This report has not yet produced the foundation Python receipt,
a role-specific weight edit, or a behavior qualification. The route index
does not claim any of those events occurred.
