# MC10D Qwen successor v1.9.0 pre-execution boundary

Date: 2026-09-07

Canonical main remains frozen at `0abaed85873c3f8de04765847eb7700b0e20433f`.

## Why this package exists

The owner requested that every MC10D step that can safely be combined should be combined, while preserving the existing gate order and Elaina-fidelity doctrine.

The workflow allows one consolidated **local pre-pointwise transaction** now, but it does not allow private pointwise and full simulation to be bundled into the same blind run. Private pointwise must be bound to the exact successor pointwise-ready bundle and governance receipt hashes created by this step. Full simulation/falsification must then be bound to the exact frozen pointwise result.

Therefore v1.9.0 does all safe deterministic work through the pointwise authorization boundary and then stops.

## Release

- package: `ALICE_MC10D_QWEN_SUCCESSOR_REFREEZE_BREADTH_v1.9.0.zip`
- package SHA-256: `5A83B08399A8AB8A06FAE0DCA7345E87D2355544FC593E7207FDAF17AC97FDD0`
- launcher: `Start-ALICEMC10DQwenSuccessorRefreezeBreadthV190.ps1`
- launcher SHA-256: `8ABF9EEB5BF022DEA701AA4F9A007FAFE61E19FCA5A16DEA2267D1A86374E169`
- build receipt SHA-256: `37AC04CFAAF7BB9A58D152D352C80063AFACB5621DADCBA7C87B4F0BF0D59AE1`
- controller SHA-256: `271C0C63A81F0B3B3372F352C7F25E1CD58CE01C938986444E6999AC2F87A173`
- selftest SHA-256: `6DC26D664AE2EB4202FE349EB30D9EC8FD59466F1D90F423BA440A94B4D1BC89`

The ZIP passed CRC, deterministic byte-identical rebuild, clean extraction, Python compilation, and fresh-extraction selftest.

## Authority inputs

- MC10D freeze: `22B0ADBCCF442B0B3654F964E35AB77A044AF7623891E2A60C72B06A94ECE9A3`
- v1.7.2 scientific parent: `4D9EE7FE7AACB510A1170B5C53B9E9F07D1182FFDC04EE55FA2A533097324533`
- nested v1.7.0: `66128E4A11512B097BC7F0B394126CE90A4464051B72950DE2B107C425FE934E`
- exact v1.7.0 deterministic refreeze controller: `936DD235CC2BF486A7CD830446594677FDDA6ADE687B700D5B5144F9E4BFE74E`
- verified Qwen result ZIP: `d5f9f75d33ca92d7455677ba4f04ca8fa090ccba03b3946732671a0403669d53`

## Exact safe batch

v1.9.0:

1. independently revalidates all 84 Qwen evidence-manifest files and recomputes the public score;
2. recovers and validates preserved Gemma decision-centric evidence without rerunning Gemma;
3. verifies the exact frozen Mistral and Granite identities;
4. installs a no-inference Qwen evidence adapter;
5. patches only the exact historical v1.7.0 judge-binding boundary to replace the failed GLM family with the owner-approved Qwen successor;
6. disables the historical pending-public-judge remote branch;
7. runs the historical deterministic refreeze on a disposable clean clone of canonical main, not the owner's dirty/untracked worktree;
8. requires the resulting bound family set to be exactly Gemma + Qwen + Mistral + Granite;
9. requires effective pool = 287 and deferred = 1;
10. requires pointwise start = true and full simulation start = false;
11. writes a new truthful successor binding rather than editing failed v186 into success;
12. migrates the previously owner-accepted controlled-synthesis breadth prerequisite to the new successor hashes;
13. emits the exact private-pointwise dispatch authority.

It does **not** submit Kaggle/Magnolia inference.

## Judge identities

- Gemma: `gemma4:31b-it-q4_K_M` / `6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7` / thinking on
- Qwen: `qwen3.8:27b-q4_K_M` / `25b843619e944cd0ae6069f94ff4e5e26a16e109ccbc0a66a0f05979ed70098e` / exact qualified Kaggle 2x-T4 profile
- Mistral: `mistral-small3.2:24b-instruct-2506-q4_K_M` / `5a408ab55df5c1b5cf46533c368813b30bf9e4d8fc39263bf2a3338cfa3b895b`
- Granite: `granite4.1:30b-q4_K_M` / `3f3e5df8a021439fd6f867a0e526bdc303cac79c811201cb6bac193298cb9fcd`

Qwen's known public-calibration errors Q09 false-HOLD and Q12 false-REJECT remain explicit challenge debt. They are not post-hoc tuned away.

## Fidelity and synthesis invariants

- E0 is never generated.
- E-INF remains uncertain historical hypothesis and may remain UNKNOWN.
- A-SYN is broader behavioral completion but cannot fabricate source history or lived memory.
- high-impact future A-SYN floor = 3 independent E0 families;
- 4+ independent E0 families = strong-anchor tier;
- current 287 pool is not mutated by the breadth migration;
- novel compatible behavior gets extra challenge, not automatic rejection;
- repairable scope/role/context defects require a new ID and full retest;
- all 18 falsification families remain;
- all 8 zero-tolerance vetoes remain;
- synthetic ancestry/volume cannot increase historical Elaina truth confidence.

## Frozen stop after successful v1.9.0

Expected outputs:

- `ALICE_MC10D_POINTWISE_READY_QWEN_SUCCESSOR_v1.9.0.zip`
- `ALICE_MC10D_POINTWISE_DISPATCH_AUTHORITY_v1.9.0.json`

At that point:

- four-family successor binding = created;
- breadth prerequisite = migrated;
- private pointwise = authorized but not started;
- MC8 = sealed;
- full simulation = blocked;
- A-SYN acceptance/promotion = false;
- training = false.

The next executable must bind the private blinded pointwise screen to the exact hashes above. Only after that screen freezes may the 18-family / 8-veto full simulation-falsification package be dispatched to finish the MC10D scientific decision boundary and hand MC10E its SELECT / ABSTAIN input.
