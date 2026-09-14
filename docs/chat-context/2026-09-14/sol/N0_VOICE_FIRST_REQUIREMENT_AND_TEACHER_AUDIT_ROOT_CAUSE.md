# N0 voice-first requirement and teacher-audit root cause

Date: 2026-09-14

Owner requirement:
- A.L.I.C.E. will be used primarily through voice.
- EIPM/personality must control not only lexical choice but tone, emotion, warmth, seriousness, playfulness, intensity, pace, emphasis, hesitation, vulnerability, teasing, and turn-taking.
- Later Elaina-derived expressive behavior should use governed Elaina evidence rather than generic stereotypes.

Boundary:
- N0 stays public/non-identity and learns generic spoken-expression reasoning.
- Elaina-specific expressive tendencies belong to N1/N2 and remain behind private identity authorization.
- EIPM outputs expressive policy; replaceable TTS renders it. Speaker-timbre cloning is separate.

N0 changes on alice-eipm-v1-build:
- voice-first contract added
- 8 VOICE competencies added as supplemental public foundation
- separate eval-only voice readiness base added
- 40-row v0.4b public voice teacher shard added (32 train, 8 dev)
- teacher bank registry v0.4b now expects 383 rows / 51 competencies
- N0 v0.2 config now requires voice readiness in addition to original fixed suite

Teacher audit failure:
- first audit failed before reading curriculum with `ModuleNotFoundError: alice_personality`
- root cause: host wrapper exported PYTHONPATH but magnolia_udocker_exec.sh did not forward it into the container
- concrete fix: PYTHONPATH is now forwarded through the existing env list
- do not create an infrastructure loop; rerun one upgraded v0.4b audit only

No GPU used. No private identity gradient authorized.
