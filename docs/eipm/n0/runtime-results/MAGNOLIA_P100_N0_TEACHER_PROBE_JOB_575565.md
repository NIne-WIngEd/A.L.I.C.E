# Magnolia single-P100 N0 teacher diagnostic — job 575565

Date: 2026-09-14

## Result

Job `575565` completed the step-1000 public N0 teacher diagnostic on one Tesla P100. The backbone checkpoint was not modified. No private identity gradient was used.

Checkpoint under evaluation:

- model: `alice-n0-semantic-v0.1`
- step: `1000`
- total tokens seen: `16,384,000`
- checkpoint receipt SHA-256: `6ec8f977032fcaa225a5ecb92f01b17caa7b364b7a033512fce263b7cf34c03a`

## Robust held-out MLM evidence

The diagnostic reused the true pre-existing dev documents under eight deterministic masking patterns instead of changing the corpus split after training.

- mask repeats: `8`
- batches total: `16`
- masked-token exposures: `3,960`
- visible-token exposures: `26,528`
- mean masked-token NLL: `7.886467143622312`
- masked-token perplexity: `2661.026271237024`
- repeat NLL standard deviation: `0.21838322864697152`

This is a more stable estimate than the original one-mask step-1000 result while preserving the same genuinely held-out document set.

## Frozen-backbone Sol semantic probe

The ModernBERT backbone was frozen. Only the generic candidate-ranking head was trainable.

- trainable parameters: `404,097`
- governed train examples: `43`
- governed dev examples: `43`
- train competencies represented: `43`
- dev competencies represented: `43`
- best saved dev top-1 accuracy: `0.6744186046511628` (`29/43`)
- best saved dev supported-set separation: `0.6744186046511628`
- best saved dev mean margin: `0.18771825817435286`
- dev failures: `14`

The 14 missed competencies were:

`ALIGN-02`, `ALIGN-03`, `ALIGN-05`, `EPI-02`, `EPI-04`, `PRAG-01`, `PRAG-04`, `RANK-01`, `RANK-04`, `SEM-03`, `SEM-04`, `SOC-02`, `SOC-05`, and `TEMP-01`.

The probe shows useful information is already linearly/nonlinearly extractable from the frozen step-1000 backbone, but the existing curriculum is too sparse to support robust generalization. This is especially clear from the training trajectory: the frozen head eventually reached 100% on the 43 training examples while epoch-30 dev accuracy fell to about 53.5%. The correct response is not to unfreeze the 352M-parameter backbone on 86 tiny examples.

## Interpretation

This diagnostic closes the question of whether to blindly continue MLM or immediately apply full-backbone curriculum training. Neither is justified yet.

The efficient next experiment is failure-driven public teaching with the backbone still frozen:

1. preserve the original 60-row Sol seed and 26-row coverage shard unchanged;
2. add a small repair shard only for the 14 observed failure competencies;
3. evaluate the previous frozen probe on the expanded dev set before repair training;
4. train a fresh frozen ranking head on the expanded train set;
5. compare on the exact same expanded dev set;
6. only if repair examples fail to improve transfer should additional MLM pretraining become the next likely bottleneck.

The repair shard is `training/eipm/n0/sol_curriculum_repair_v0.3.jsonl`. It adds two train examples and one new dev example for each failed competency: 42 rows total, all public/generic and Sol-authored under owner authorization.

## Warning classification

The Transformers `UNEXPECTED` keys (`head.dense.weight`, `head.norm.weight`, `decoder.bias`) are expected when loading an MLM checkpoint into the backbone-only `ModernBertModel` used by the ranking head. They do not indicate checkpoint corruption.

The Magnolia Linux 3.10 kernel warning remains an environmental risk already known from prior runs. Job 575565 completed normally, so it does not justify reopening infrastructure qualification.

No private E0/E-INF/A-SYN gradient is authorized by this result.
