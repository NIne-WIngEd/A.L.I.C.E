# Proposed Qwen fallback activation

This is a reviewable amendment, not an installed policy or a ready remote-run launcher. No model is downloaded, no scheduler job is submitted, and no new judge is bound by these files.

The source `mc10d_judge_binding_policy_v1.json` fixes Gemma/GLM/Mistral/Granite for the existing freeze. It names Qwen as the next fallback and says fallback activation requires an explicit later amendment. The accepted general audit direction did not itself name a replacement for that frozen family set.

The proposed amendment permits one public Qwen qualification attempt under a declared profile and compute budget. A subsequent verified binding would replace GLM with Qwen while keeping four independent families. It preserves the public scoring rule and current candidate pool. It grants no private judging, acceptance, promotion or training authority.

The complete model digest and compatible installed runtime are still preflight inputs. The public website exposes a short identifier; that is not substituted for a complete digest. This proposal permits resolving and freezing the exact catalog artifact before any inference, subject to the named tag and quantization. An unresolved or drifting artifact stops execution.

`validate_fallback_amendment.py` verifies the draft against the exact source policy, public task file and effective scoring contract. It also checks that the preserved live state has not been declared successful. `VALIDATION_RECEIPT.json` records the offline checks. These checks establish consistency of the proposal, not owner ratification or model quality.

Astra's recommendation is to activate this narrow fallback route. The owner decision must be recorded explicitly before a launcher can ratify or run it. If it is accepted, the next implementation should use the existing provider contracts and a bounded CPU preflight. Do not produce another chain of edits that masquerades Qwen as a successful GLM/v186 run.
