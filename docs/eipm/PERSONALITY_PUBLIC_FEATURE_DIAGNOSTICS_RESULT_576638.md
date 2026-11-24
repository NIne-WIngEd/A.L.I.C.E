# Actual frozen-feature diagnostic: Magnolia 576638

This no-fit job completed `0:0` in 21m 41s under `mxrayan`, at immutable tested
code `8a986eeacab71698d54090929350d929c138864b`. It used the separately closed
152-input public feature package and original learned export, with CPU/four
threads, BF16 features and FP32 readout in Torch `2.7.1+cu118`. It loaded no
publisher checkpoint or tokenizer, performed no new publisher forward or
training, installed nothing and opened no private identity or FINAL input.

Original learned and fixed metrics reproduced exactly before the controls.
Complete feature and first-party parameter bytes were rechecked afterwards,
including a fresh full package import. Independent retrieval verified original
plan/export/import/code/runtime bindings and both file/canonical seals. Root
and independent reviewers replayed per-example winners, correctness, ranks,
margins, all 18 aggregate cells, family means and replacement/familiarity counts;
cross-entropy arithmetic also agrees within FP32 rounding.

| Evidence | Exact file SHA-256 | Canonical receipt SHA-256 |
| --- | --- | --- |
| Complete diagnostic | `adb64dcce4b30c8a7c8636e95f7648bb447c4dd9c5597249ac6461909031a33c` | `be7dc51ae515c100dac5ea40db31a642f622caead6963c3017c4f254d5891e9b` |
| Declared diagnostic plan | `e9075d448eed83a7f3ea80f66abb5a030b33f66c85ee38fa1cd223a49306c08b` | `7ab28f4e8f1df870ce0a7865218525038c7dc451d8586166781c7dff377b9d91` |

The byte-identical evidence is preserved under
`evaluation/eipm/gemma_n0/receipts/`. The complete result retains every public
example's logits, feature/token/cache links and original/control outcomes;
summary tables do not substitute for those observations.

| Scorer / source | Retained TRAIN target hits | Retained DEV target hits | Retained sentence-unseen DEV target hits |
| --- | --- | --- | --- |
| Learned / original | 9 / 56 | 1 / 16 | 1 / 14 |
| Learned / constant TRAIN source | 3 / 56 | 1 / 16 | 1 / 14 |
| Learned / within-split cyclic source | 1 / 56 | 2 / 16 | 2 / 14 |
| Fixed / original | 15 / 56 | 4 / 16 | 3 / 14 |
| Fixed / constant TRAIN source | 5 / 56 | 1 / 16 | 1 / 14 |
| Fixed / within-split cyclic source | 4 / 56 | 1 / 16 | 1 / 14 |

Source substitutions are destructive ablations with unchanged old annotations,
not reviewed counterfactual truth or entity-role reversal. The two cyclic DEV
hits are accidental retained-target matches, not improved correctness for those
new sources. Constant-source selection and cyclic ordering are target-independent.

The learned readout materially depends on source features. Constant substitution
changes 40 of 55 actually replaced TRAIN winners and 11 of 16 DEV winners;
cyclic substitution changes 37 of 56 and 10 of 16. Mean DEV JS divergence is
0.042741 / 0.054178. These observations rule out a wholly source-independent
choice rule on the checked examples, but do not establish correct semantics.
The fixed control changes 12 / 11 DEV winners with JS only about `7e-7`: its
near-tied rankings mean winner changes alone exaggerate probability sensitivity.

Every DEV target description is absent as a TRAIN positive. The learned readout
chooses a familiar TRAIN-positive description on 15 of 16 original DEV rows;
the fixed control does so on 11. Uniform selection from these actual candidate
pools would choose a familiar description with expected fraction 0.78125.
All TRAIN pool descriptions are familiar, so 56/56 familiar TRAIN choices is
tautological. This is a concrete, confounded association to investigate, not
proof of a unique causal shortcut or inherited Gemma personality.

The only learned original DEV hit is in the two-row, three-candidate stratum;
there are no learned hits in the sampled 7/12/48-candidate DEV strata. Different
pool composition and tiny counts do not establish a cardinality ceiling. Slot
strata give no persuasive position pattern. Mean target cross entropy improved
relative to both the untrained architecture and fixed control despite weak
argmax accuracy; the report must retain both optimization and ranking findings.

The learned 49-state mixture is global and query-independent, with effective
states 48.99704, entropy 3.8917599 versus uniform maximum 3.8918203, and weights
between 0.0200845 and 0.0208135. This run barely distinguished depths. It does
not establish successful role-sensitive layer selection or justify discarding
middle layers. These observations concern this public semantic readout, not
the separately initialized governing N1/N2/N3 identity stages.

The owner explicitly requires research before choosing a repair. Primary
research on transfer to unseen relations, frozen-feature learning and shortcut
evaluation is being assessed against these observations and Alice's N0 role.
No training recipe, bigger architecture, publisher weight edit or acceptance
decision follows from this diagnostic alone. A justified next experiment must
use independently grounded gold and matched controls and keep DEV identified
as observed model-selection data. Full role qualification, practical measured
inherited-judgment suppression, N1 conditional identity meaning, complete N2
judgment and N3 calibration remain separate requirements. N0 is unqualified.
