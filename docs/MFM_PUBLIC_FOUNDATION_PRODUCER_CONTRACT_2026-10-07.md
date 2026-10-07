# Full MFM foundation producer contract — 2026-10-07

Construction contract reviewed for implementation. Training allocation and complete bank pin remain pending; no learned capability or qualification is established by this document.

The next producer should be a versioned fresh public conditional foundation checkpoint that continues into the full MFM producer. E2E supplies real learning material for that construction, but does not supply complete formation semantics. This follows `docs/MFM_FULL_ROLE_SEQUENCE_SUCCESSOR_2026-10-07.md` and the current fresh-source-only route. The full 4,862-source export is the intended roster; scheduling stops are resumable execution details, not a smaller capability program.

## Actual artifact, input, output and initialization

Use the complete 4,862-source bank, the 36,597 training pairs and 4,271 language-development pairs. Join each pair to its source by exact processed source IDs and `source_key`, retaining semantic-group and original-byte ancestry. The external census pin is `0885f9a52b8ea967127a5a6b7708331163b214d3dea81ed598d24c57de6df27d`; the completed bank must have its own separately verified pin. Inputs are detached BF16 `[source_length,3840]` states plus masks. Outputs are the prepared canonical `{"value_text": ...}` sequences: decoder inputs `[BOS,*body]`, labels `[*body,EOS]`, with BOS 2/EOS 1. Human targets never enter Gemma feature extraction.

Preserve width 768, six decoder layers, 12 attention heads, the full 262,144-token tied fresh embedding/output matrix, fresh source projection and positional embeddings. Use the pinned `fresh-token-position-scale-0.02-v1` construction and save its seeded, untrained state before updates. Keep full formation sequence capacity; E2E's maximum 84 target tokens does not teach the later positional range. Do not silently load the failed formation checkpoint or reuse the old `source_reconstruction` task as meaning-to-text: its task registry and initialization describe a different objective. Future multitask binding requires an explicit version.

A training manifest must bind material and encoded-pair hashes, census seal, complete bank/export seals, prepared clone/processor/template/tokenizer bindings, source roster and split ancestry, config/init/seed/code, objective and sampler, optimizer/schedule/budget/runtime, and resumable checkpoint cursor. Exact continuation keeps those immutable recipe bindings; a changed learning objective is a new stage with explicit parent lineage.

## Public conditional objective

Complete-sequence teacher-forced cross entropy, including EOS, is directly supported by the current decoder. The recommended objective gives each semantic MR equal total mass and distributes that mass across its retained alternative references. This prevents variable reference counts from masquerading as independent experiences. Freeze exact token normalization in the training manifest rather than assuming another 50/50 mixture.

E2E references are not complete grounded truth. The original paper allowed content omission; its 50-pair sample reported 22% omitted and 18% added content. Those are not measured rates for our prepared corpus. Retain the references as conditional language teaching while measuring additions and omissions separately; do not use their likelihood as a formation-faithfulness verdict. Source: [E2E, section 3 and Table 5](https://aclanthology.org/W17-5525.pdf).

The actual census has only 3,341 positive target token types, about 1.27% of the full vocabulary, with 2,349 occurring fewer than ten times. This is concrete conditional training, not evidence of sufficient general sequence coverage.

## Transfer into the full semantic formation producer

Continue the learned fresh weights through three distinct learned responsibilities:

1. **Reference selection.** Score caller-provided candidate representations from the decoder state using reference type and eligible output path; supervise the selected candidate index. Resolve it through the externally bound inventory and keep catalog digests outside learned output. Candidates come from the independently frozen caller manifest, never the teacher decision. Source-declared subjects retain their narrower eligibility. Disposition scopes remain extensible semantic labels; the older catalog's scope-candidate assumption must not override the newer caller contract.
2. **Literal copying.** Learn selection over eligible observed source positions/spans independently from reference identity selection. The missing dependency is an exact map connecting processed tokens, original UTF-8 boundaries, structured/native locators and canonical output encoding. Plain source IDs are insufficient where escaping or normalization changes token correspondence. Repeated occurrences and multiple valid alignments need explicit treatment.
3. **Semantic authoring.** The fresh decoder continues to learn warranted `value_text`, proposals/dispositions, subject, epistemic status, time, uncertainty, sensitivity, episodes, corrections/deletion, norms/skills, missions/outcomes and self/relationship distinctions.

Typed selection has a concrete precedent in [RAT-SQL, sections 4.3–4.4](https://aclanthology.org/2020.acl-main.677.pdf); that paper also reports failures of simple token pointers on multi-token values. The [pinned Fairseq pointer-generator, lines 269–333](https://github.com/facebookresearch/fairseq/blob/3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99/examples/pointer_generator/pointer_generator_src/transformer_pg.py#L269-L333) learns a generation/copy gate and aggregates attention through actual source-token identities. These mechanisms support the separation above, not a claim that copying teaches meaning.

## Foundation before personalization

Shared foundation material teaches formation competence without an owner biography. Full-role fictional worlds, observations and authorized prior-registry snapshots must be frozen before labels, with diverse positive semantic transitions and contrasting cases. Broader rights-admitted conditional language material remains missing; E2E cannot fill that gap. Private instance adaptation remains separately governed.

The compiler may allocate new local bookkeeping and resolve predicted selections losslessly. Claim still owns writes and current authority checks. Native modalities and accepted-memory → judgment/action → outcome/revision learning remain required construction.

## Resources and capability rejection conditions

The complete bank plans 1,848,875,520 bytes of states before custody overhead. Measure actual export throughput, setup cost, peak RSS and longest-source performance. Measure the intended producer's full backward and optimizer step, source loading, vocabulary-projection cost, peak memory, checkpoint size and exact restart. FP32 weights, gradients and two Adam moments for approximately 267M parameters alone are about 4.27 GB; activations and temporary logits are additional. Freeze device, precision, batching, optimizer, schedule and update budget from those measurements. Scheduling stops must resume the same full roster and objective.

Lower CE or valid JSON is insufficient if changed input names, polarity, ratings or relations fail to change generated meaning appropriately. Use paired source interventions; rank/probability changes with unchanged wrong argmax still demonstrate some source effect. Later reject selector claims if renaming/reordering or same-type distractors change resolved semantic choice. Reject copy claims if they select template text, wrong repeated occurrences or invalid byte boundaries. Successful reference emission with incorrect uncertainty, temporal evolution, correction/deletion or outcome semantics cannot establish full MFM capability.

The public objective is now fixed as sequence mean including EOS, reference mean, then equal semantic-group mean; epochs shuffle the complete immutable training roster deterministically. The unresolved decisions are the measured training allocation, candidate serialization/representation, pointer-versus-action wire format, exact copy alignment, and the next broader admitted teaching material. None requires adopting a pretrained head or lexical bridge.

## Inspected local contracts

- `docs/MFM_FULL_ROLE_SEQUENCE_SUCCESSOR_2026-10-07.md`
- `docs/research/MFM_SEQUENCE_ROUTE_COMPARISON_2026-10-07.md`
- `docs/research/MFM_NEXT_ROUTE_DESIGN_REVIEW_2026-10-07.md`
- `docs/research/MFM_CONDITIONAL_TEACHING_MATERIALS_2026-10-07.md`
- `docs/receipts/MFM_CONDITIONAL_CENSUS_576995_PUBLIC.json`
- `scripts/mfm/preflight_public_conditional_material.py`
- `scripts/mfm/export_public_conditional_features.py`
- `scripts/mfm/formation_caller_input.py`
- `src/cognitive_kernel/formation_v1_specialist.py`
- `src/cognitive_kernel/formation_decision_specialist_v16.py`
- `src/cognitive_kernel/formation_sequence_specialist.py`

The native-model-building, research-FBM and mandatory diagnosis skills were read. Prior failures remain preserved; this note does not revise their verdicts.
