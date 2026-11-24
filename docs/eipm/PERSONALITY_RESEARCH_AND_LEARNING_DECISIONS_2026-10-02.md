# Alice personality: research and learning decisions

This records construction decisions, not a learned personality result or N0
approval. Full Alice and full Fable v1 FBM remain the scope; FloRA is excluded.
The exact 24 GB pretrained Gemma remains frozen. Custom N0 and the retired
weight modification program stay outside v1.

The owner's latest clarification controls the design: complete suppression of
Gemma costs too much. V1 requires practical best-effort suppression of observable
judgment interference, semantic retention and an honest record of residual
influence. Latent pretrained associations do not necessarily govern a judgment.
Qualification uses reviewed tolerances, not an invented zero-influence condition.

## Source of the design

Alice's EIPM role/voice contracts at
`alice-eipm-v1-build@021c5021a98104b35f9c8e94b19e48d21f25f132`, and the complete
Fable comic/release documents at `1805a01c73575378246cac8cbbb72cd891e8528e`,
remain authoritative specifications. Research branches and refreshed Graphify
context locate relevant material; original files and primary publications
support conclusions. An index association is not a trained concept.

The comparison includes recent work and older foundations. It does not claim a
globally best architecture or transfer of published scores to Gemma 4. Published
characters and ordinary user profiles are not Alice's identity gold.

## Useful methods and their limits

| Primary source | Useful method | Decision for Alice |
| --- | --- | --- |
| [Persona Vectors, Anthropic, 2025](https://www.anthropic.com/research/persona-vectors) | Contrastive activations diagnose trait drift; some post-hoc steering incurs capability costs. | Use matched contrasts as diagnostics; establish causal changes in our judgments before repair. Do not import foreign-model vectors or upstream weight interventions. |
| [The Assistant Axis, Anthropic, 2026](https://www.anthropic.com/research/assistant-axis) | Persona structure occurs in pretrained base models as well as assistants. | NON-IT/headless does not prove neutrality. Test irrelevant persona framing. Their default-assistant stabilization objective is not Alice's objective; blindly subtracting that direction is unsupported. |
| [PersonaGym, EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.368/) and [author code](https://github.com/vsamuel2003/PersonaGym) | Dynamic environments, multidimensional behavior assessment and evaluator separation. | Use reviewed Alice-relevant conditions and independent evaluation. Generated scenarios/rubrics remain proposals; general benchmark scores do not qualify source authority or IDP. |
| [InCharacter, ACL 2024](https://aclanthology.org/2024.acl-long.102/) | Open-ended interviews supplement direct self-report scales. | Add interviews to behavioral cases and human review. Trait summaries do not replace conditional policies, values, relationships, source history or uncertainty. |
| [Character-LLM, author implementation](https://github.com/choosewhatulike/trainable-agents) | Reconstructed scenes and protective examples teach character behavior. | Adapt conditional coverage and anti-fabricated-history tests. Synthetic reconstructions cannot become history or Alice experience. Whole-generator training and deliberate knowledge-forgetting are incompatible with this route. |
| [RoleLLM, ACL 2024](https://aclanthology.org/2024.findings-acl.878/) | Separates role knowledge construction from speaking style. | Keep evidence and expression targets distinct but connected. Catchphrases cannot substitute for EIPM judgment; borrowed roles cannot define Alice. |
| [PersonaPlex, NVIDIA, 2026](https://arxiv.org/html/2602.06053) and [author code](https://github.com/NVIDIA/personaplex) | Separate role/voice conditioning with actual full-duplex interaction and timing evaluation. | Separately observe voice identity, expressive intent and timing; jointly review outcomes across replaceable renderers. A new prompted speech generator does not replace approved Gemma/EIPM ownership. |
| [Moshi, Kyutai author code](https://github.com/kyutai-labs/moshi) | Actual speech/text interaction carries information beyond transcripts. | Preserve uncertain acoustic context and test actual speech. Text descriptions of audio and numeric intent controls do not qualify complete spoken personality. |
| [ArmoRM](https://arxiv.org/html/2406.12845) and [author code](https://github.com/RLHFlow/RLHF-Reward-Modeling/tree/main/armo-rm) | Multiobjective linear heads with missing-rating masks above fixed embeddings, then contextual gating. | An economical N2 baseline; preserve dynamic Alice dimensions, source roles and full packet instead of importing generic axes or collapsing everything to its final scalar. |
| [PAL](https://proceedings.iclr.cc/paper_files/paper/2025/file/2858f8c8683aaa8c12d487354cf328dc-Paper-Conference.pdf) and [author code](https://github.com/RamyaLab/pluralistic-alignment) | Small conditional mappings learn preferences on frozen representations. | Compare a small conditional head with the current graph/token/latent model using equal features and budgets. Host/user localization is not source-person authority or a complete IDP. |
| [RewardUQ, February 2026 preprint](https://arxiv.org/abs/2602.24040) and [author code](https://github.com/lasgroup/rewarduq) | Frozen-feature head ensembles and Bayesian linear heads separate accuracy from probability/confidence calibration. | Compare practical first-party uncertainty heads against N3 temperatures if observed errors justify it. Exclude upstream LoRA/DPO variants; ensemble agreement may share the same source error. |

These are methodological adaptations. No third-party code, checkpoints or
datasets were imported. Check code, model and dataset licenses separately before
future reuse; repository code licensing does not authorize every weight/data file.

Explicit A/B/tie likelihoods and soft preference ensembles are useful only when
reviewed targets distinguish indifference, co-validity, population disagreement
and unavailable review. [Pairwise Calibrated Rewards](https://proceedings.neurips.cc/paper_files/paper/2025/file/53dbd7e34fab703a639964e2d3ee9e84-Paper-Conference.pdf)
offers a comparator, not Alice gold. Selective/conformal methods such as
[SCOPE](https://arxiv.org/abs/2602.13110) are later experiments: shared source
derivatives, family shifts and dynamic candidate sets make naive exchangeability
or coverage claims invalid. Strong generic reward results do not establish
source identity, voice or owner fidelity.

The research review also checked prior multi-view code through Graphify at
`f357a767ca1542a4043944b380663f27b50c33ec` and read later doctrine/research files
directly at `021c5021`. Old 640-width custom-N0 geometry and teacher sequences
are superseded. Search accounting: 22 requested results across persona/latent
influence/voice angles and 40 across preference/plurality/calibration/source
conflict/readout angles. These 62 result slots include duplicates and versions;
they are not 62 independently reproduced studies. Primary articles, papers and
author repositories were read; no reported model result was reproduced here.

## Required data-to-role design

| Component | Governing purpose | Teaching facts and current substantive gap |
| --- | --- | --- |
| N1 | Source-supported concepts, conditional policies, support/disconfirmation and inspectable residual nuance; history/inference/synthetic/host/relationship/self distinctions. | Reviewed applicability conditions, exceptions, expressive tendencies, supporting and contradictory evidence, uncertainty and identity-versus-drift cases. Current alignment/provenance losses are auxiliary mechanics, not the complete objective. |
| N2/EIPM | Contextual preferences and co-valid alternatives; stance, values, relationship/emotion, communication, expression, uncertainty, contraindications and grounded rich identity state. | Candidate-bearing situations and independently reviewed sparse full-IDP targets. Source rows do not automatically supply preferred indices, negative alternatives or full head labels. Unlabelled facts stay unavailable. |
| N3 | Calibration, ambiguity, margins, failure tails and owner-reviewed fidelity, preserving source authority. | Separate calibration families and reviewed choice/head/pair/risk/voice-confidence targets. Positive temperature fitting cannot reverse wrong order; substantive mistakes require N1/N2 repair. |
| Voice/consumers | Native judgment and expressive intent must govern words and delivery across replaceable providers. | Actual spoken review, cue confidence, lexical/delivery alternatives, timing/emphasis mapping and response enforcement. Current text packaging/numeric controls are interfaces, not complete integration. |

The current core is a trainable scaffold. Administrative candidate masks are not
personality preferences. Keep permissible but uncharacteristic actions admitted
when teaching characteristic choice. Initiative, clarification and principled
disagreement cannot be chosen if the candidate producer omits them.

N1 provenance prediction sees supplied source-kind metadata; accuracy can be a
shortcut. Its concept states receive support-graph messages before alignment
scoring. Compare metadata-only, graph-only, text-only, withheld-edge and wrong-edge
controls without altering historical permissions to manufacture semantic
negatives. Inspectable concepts may remain outside weights; supplying a persona
bank is not proof of learned judgment. Compare explicit N1 freezing with reviewed
N1 continuation during N2 after actual failure diagnosis, rather than declaring
either correct by architecture alone.

Evidence pointer scores are relative distributions, not support assertions.
Eligible but irrelevant sources can receive mass when no source is relevant.
Test sufficiency/abstention interpretation against decisive-source removal and
unrelated-source retention before any consumer treats a pointer as justified.

## Practical influence diagnosis and repair

Consume the precommitted public semantic experiment first. FewRel measures
relation matching, not personality preference. Its global layer mixture also
differs from the identity core's query-conditioned layer scoring.

If a material failure warrants more measurement, compare neutral and social
framing with appropriate token/position controls. Separately alter relevant
rules/evidence/conditions. Prefix sensitivity alone cannot distinguish persona
interference from extra-token effects or appropriate sensitivity to meaning.

For native judgment controls, use explicitly authored public fictional rules
and independently reviewed targets first. Hold evidence fixed while changing
flattery/prestige/style; hold style fixed while changing decisive evidence.
Include co-valid choices, missing support, order permutations and source/host/self
confusion. These rules test the mechanism and never become Alice identity gold.

Repair measured interference through reviewed TRAIN consistency examples or the
first-party readout/conditional path. Compare held-out judgment and semantic
retention against the original baseline. Stop costly escalation without material
benefit and report residual influence. No upstream gradient, publisher weight
editing or universal persona erasure is planned. Numeric tolerances need actual
evidence and independent review; this record invents none.

## Evidence and compute

Actual publisher custody, full-layer forwarding and public source admission ran
on Magnolia under the owner's account. Public experiment 576600 completed with
weak unqualified learning; diagnostic 576638 independently reproduced it and
measured source dependence, familiar-description preference and near-uniform
depth weights. Repair-specific primary research and a precommitted freshly fitted
candidate-only control are recorded in
`PERSONALITY_SEMANTIC_REPAIR_RESEARCH_AND_CONTROL_PLAN_2026-10-02.md`.
No winning learning repair or N0 approval follows from those findings.
Exact isolated candidate diagnostics identified the
compiler's incorrect global-context/substr-name assumptions. The repair preserves
declared lanes/masks and rejects explicit exclusion of known positive losses;
successful compilation still does not accept source content or authorize training.

Portable public feature custody requires a completed experiment, original
unchanged receipts, complete banks and producer-side source closure. Import
records a separate consumer binding; fixed inputs cannot answer new raw prompts
or establish neutrality. Actual handoff 576635 completed, independently closing
all 152 complete public banks and a separate CPU consumer. The importer uses CPU
RAM. No CUDA trainer, Kaggle job or GPU resume proof has run.

Magnolia and Kaggle remain the first compute routes. Paid GPUs are the last
fallback after concrete free-route failures and a concrete workload/cost choice.
Separate GPU memory spaces are not a single pool. Public closed features can
avoid moving the 24 GB publisher checkpoint; they do not establish private
execution isolation.

FBM records source/data-to-role diagnosis, justified choices, causal controls and
bounded repair. Fixtures, scaffolding and command logs are procedure evidence,
not learned Alice personality or FBM transfer qualification.
