# Current continuation — Qwen a3 ready after verified publisher recovery

Read [STATE](STATE.json), the [a3 release](QWEN_TELEMETRY_SUCCESSOR_V105.json),
the [verified repair](TELEMETRY_PUBLISHER_REPAIR_RESULT_REVIEW.json), and the
[master handoff](ASTRA_MASTER_CONTINUATION_HANDOFF.md).

The shared publisher repair is complete. Magnolia's actual Git 1.8.3.1 passed
all seven publisher checks. GitHub commit 7dfff68 added only the two missing
terminal metadata files. Both publication calls succeeded. The identical retry
created no commit. Original a2 evidence and the Windows controller stayed unchanged.

Next: save the v1.0.5 ZIP and Start-ALICEAstraQwenQualificationV105.ps1 in Downloads
and run the launcher. This starts one separately identified execution,
alice-qwen38-a3-750881d854fa, under the existing public calibration approval.
It checks both closed parent runs and the exact installed publisher before
submission. Native Windows execution and actual a3 submission remain pending.
Do not rerun v104, reset a2, or alter historical results to clear a pending flag.

The existing worker, runtime pin, approved profile and 17 request hashes remain
unchanged. A3 reuses a2's frozen model manifest. Magnolia remains CPU-only:
20 CPUs, 48 GiB exclusive, node/normal, six hours. The 128-token probe must pass
the original worst-case budget gate before the 16 calibration tasks. No probe
or task retries, automatic successor or Kaggle fallback are allowed.

All 12 checks passed from a fresh ZIP, including real local Git publication
through the new evidence capture and a complete synthetic worker/collector/
verifier round trip. They also cover parent mismatch, failed scheduler queries,
publisher drift, lost submission acknowledgement and slow-CPU rejection.
The publisher's raw Git journals and outer stdout/stderr travel with results.
Pre-submission/transport stops produce STOP_ZIP. A downloaded result rejected
by verification is separately retained as RAW_RESULT_ZIP. Return the printed
ZIPs and terminal transcript; do not discard an unsuccessful result.

Qwen is still NOT_EVALUATED. Its runtime, loaded model and throughput are not
established by manifest resolution or offline tests. If Qwen passes actual
public calibration, verify the exact four family/profile/digest receipts before
constructing the successor binding and migrating the breadth prerequisite.

The actual public result pointer remains at closed a2 until a verified new result
arrives. Preserve Gemma v182, Mistral/Granite, failed GLM results, 287 candidates,
63 replacements and one deferred slot. Private pointwise, breadth, acceptance,
promotion, training and MC8 remain gated. The [Phase 2 roadmap](PHASE2_REPLACEMENT_ROADMAP.md)
still requires Stage G, G+, H, I and J with their acceptance gates. Main remains
0abaed85873c3f8de04765847eb7700b0e20433f.
