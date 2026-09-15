# N0 v0.2 Voice-First Teacher Bank Audit PASS — 2026-09-14

The owner reran `scripts/eipm/n0/magnolia_cpu_n0_v02_teacher_bank_audit.sh` after the semantic-item duplicate fix.

Result: `PASS`, exit code 0.

Key results:

- registered rows: 383;
- unique IDs: 383;
- competencies: 51 total;
- core competencies: 43;
- voice competencies: 8;
- coverage gate: PASS;
- failures: none;
- fixed core eval suite: 43 rows / 43 competencies;
- fixed voice eval suite: 8 rows / 8 competencies;
- distinct v0.4+ principle tags: 50;
- private identity data: false;
- private identity gradient authorized: false;
- model training performed: false.

The full multitask row floor remains closed at 383/1000. This is expected and is not an audit failure.

The prior false positives came from comparing generic prompt stems without candidate content. The corrected audit compares the semantic item rather than treating shared instruction wording such as `Which pair is closest in meaning?` as a leaked example.

This gate is closed. Do not rerun it unless the teacher bank changes materially or a later model result exposes a concrete issue.
