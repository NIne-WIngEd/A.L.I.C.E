# N0 v0.2 structured-state preparation ready

Date: 2026-09-15

Semantic base remains `targeted-repair-v0.1/checkpoints/step-00000080`.

Build branch now contains:
- structured-state encoder: `src/alice_personality/n0/structured_state.py`
- structured-state objectives: `src/alice_personality/n0/structured_state_objectives.py`
- mechanics config: `configs/eipm/n0/n0_v02_structured_state_v0.1.json`
- mechanics tests: `tests/eipm/test_n0_structured_state.py`
- objective tests: `tests/eipm/test_n0_structured_state_objectives.py`
- curriculum compiler: `scripts/eipm/n0/build_n0_v02_structured_state_curriculum.py`
- CPU preparation runner: `scripts/eipm/n0/run_n0_v02_structured_state_prepare.sh`

Structured branch size: 1,656,064 parameters. The semantic core remains unchanged.

Objective weights:
- summary semantic alignment 0.45
- rationale compatibility 0.30
- field semantic preservation 0.15
- field-order consistency 0.10

The compiler reuses existing governed teacher-bank text verbatim. It does not write new prompt, candidate, or rationale text. The existing train/dev split and preferred-candidate labels are retained.

Next action is one CPU-only preparation run in the existing Magnolia udocker runtime. The run compiles code, executes bounded tests, compiles the structured curriculum, and prints row/split/label counts. No GPU work is part of this preparation step.
