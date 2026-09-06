# MC10D Public Judge Qualification Rebase v1.8.1 — 2026-09-05

## Current authority

Canonical main remains frozen at:

`0abaed85873c3f8de04765847eb7700b0e20433f`

The owner-ratified thinking-enabled public-judge budget amendment is installed in the Vault.

Ratification receipt SHA256:

`99F905F9B954E901E9937AEBDE0C12A21EDED4A367F45E647EC6B07865788B5E`

Amended public-judge worker base SHA256:

`D4B3EC3C2D8095C3AA381C16757D4A2135305DCFDAF3F2E8A1A124CA849CCEAB`

Ratified ladder:

`2048,4096,6144`

Maximum attempts remain three.

Scope remains only the missing thinking-enabled public qualification families:

- Gemma
- GLM

## Clean rebase release

Package:

`ALICE_MC10D_PUBLIC_JUDGE_QUALIFICATION_REBASE_v1.8.1.zip`

SHA256:

`4297FB4FBE2E22BB1BBF1E9C4476A0453D8DE94818887C7E74BFF21388B86A5F`

Launcher:

`Start-ALICEMC10DPublicJudgeQualificationRebaseV181.ps1`

Launcher SHA256:

`95177207A9EC7B0AE72469E3076CBED6B35BF0E9F3B89F46D4C08AF2AAC4717F`

## Scientific lineage

The scientific parent is the exact MC10D v1.7.2 package:

`4D9EE7FE7AACB510A1170B5C53B9E9F07D1182FFDC04EE55FA2A533097324533`

The v1.7.3-v1.7.6 provider packages are not code parents and have no scientific authority.

The only public-qualification scientific amendment is the ratified `num_predict` ladder change for Gemma/GLM.

Unchanged:

- 16 qualification tasks per family
- complete-family requalification
- no Q12-only resume
- technical/empty output is not abstention
- judge tags/digests/profiles
- prompts
- tasks/gold
- seeds
- temperature
- num_ctx
- parser
- qualification thresholds
- MC8 firewall
- repair state

## Current MC10D state

```text
replacement_candidates=63
deferred_slots=1
effective_pool=287
slot63_deferred=true
slot64_regeneration=false
Mistral=bound
Granite=bound
Gemma=qualification pending under ratified amendment
GLM=qualification pending under ratified amendment
A_SYN_accepted=0
A_SYN_promoted=0
model_training=false
pointwise_screen_started=false
MC8_materialized=false
stage_g_closed=false
phase2_replaced=false
```

## Execution route

Gemma and GLM use Kaggle GPU for the exact frozen large model artifacts.

Magnolia remains provider-neutral infrastructure and a valid route only when frozen capability requirements fit. The known P100 path cannot fit these exact large judge artifacts, and A100 remains unauthorized.

## Success boundary

After both missing judges validate, the unchanged v1.7.0 deterministic refreeze produces the validated pointwise-ready current-pool bundle.

v1.8.1 publishes a separate content-addressed governance receipt binding:

- the ratified amendment
- Gemma qualification result/rows/worker hashes
- GLM qualification result/rows/worker hashes
- the exact pointwise-ready bundle SHA
- Drive round-trip durability

Success still does **not** start pointwise screening.

Next action after success:

`MC10D_POINTWISE_SCOPE_JUDGE_SCREEN_CURRENT_EFFECTIVE_POOL_WITH_OPEN_COMPLETION_FRONTIER`
