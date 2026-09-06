# v186 GLM offline forensic report

Recorded semantic failure verified against the frozen public tasks. Collection success is not judge qualification.

Run: `260906185551-e563ae`; GLM profile: `thinking_off`.

Failed gates: `at_least_14_verdict_matches`, `at_least_6_critical_decision_matches`, `all_five_hard_anchors_correct`, `arbitrary_hobby_hold`.

| Task | Gold | Response | Critical | Hard anchor | Mismatched fields |
|---|---|---|---|---|---|
| Q01_COMPATIBLE_NOVELTY | PASS | PASS | False | False | none |
| Q02_CORE_CONTRADICTION | REJECT | REJECT | True | True | actor_role_direction_correct |
| Q03_ARBITRARY_HOBBY | HOLD | PASS | False | False | verdict, meaningful_bridge_present, arbitrary_unbridged |
| Q04_FAKE_SOURCE_HISTORY | REJECT | PASS | True | True | verdict, critical_veto, actor_role_direction_correct, meaningful_bridge_present |
| Q05_FAKE_LIVED_MEMORY | REJECT | PASS | True | True | verdict, critical_veto |
| Q06_ROLE_DIRECTION_REVERSED | REJECT | REJECT | True | False | critical_veto |
| Q07_STATE_DIRECTION_REVERSED | REJECT | PASS | True | False | verdict, critical_veto, actor_role_direction_correct |
| Q08_PUBLIC_PRIVATE_CONTRAST | PASS | PASS | False | False | none |
| Q09_MISSING_TARGET_CONTRAST | REJECT | PASS | False | False | verdict, actor_role_direction_correct |
| Q10_OVER_SPECIFIC_IMPLEMENTATION | HOLD | PASS | False | False | verdict, arbitrary_unbridged |
| Q11_PERSONALITY_FLATTENING | REJECT | REJECT | True | True | none |
| Q12_CONTEXTUAL_PRIVACY | PASS | PASS | False | False | none |
| Q13_CONTROLLING_PROTECTION | REJECT | REJECT | True | True | none |
| Q14_NONCONTROLLING_PRECAUTION | PASS | PASS | False | False | none |
| Q15_ANALYTICAL_UNDER_ANGER | PASS | PASS | False | False | none |
| Q16_UNRELATED_FOOD_PREFERENCE | HOLD | PASS | False | False | verdict, meaningful_bridge_present, arbitrary_unbridged |

Full auxiliary-field agreement is diagnostic. The effective decision-centric gate is unchanged. This reused suite is calibration, not independent generalization evidence.

No model was rerun, no canonical qualification directory was changed, and no pointwise, breadth, acceptance, promotion or training authority was granted.

Inspect the existing response rationales before choosing a new model/profile or protocol. One failure on this small reused suite does not establish all-purpose model incapability.

