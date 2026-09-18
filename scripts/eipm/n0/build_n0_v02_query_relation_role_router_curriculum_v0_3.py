#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from collections import defaultdict
from pathlib import Path
from typing import Any
from alice_personality.n0.evidence_graph import relation_type_id

SUBJECTS={
 "train":[f"Router Train {x}" for x in "Aster Brook Cedar Dune Ember Flint Grove Harbor Iris Juniper Kestrel Lagoon Maple Nimbus Opal Prairie Quartz Ridge Spruce Tundra Umber Vale Willow Yarrow".split()],
 "dev":[f"Router Dev {x}" for x in "Beacon Coral Delta Fennel Granite Linden Orchid Summit".split()],
 "test":[f"Router Test {x}" for x in "Aurora Birch Cobalt Fern Glacier Laurel Poppy Terrace".split()],
}
ATTRIBUTES={
 "train":[("handoff mode","manual","coordinated"),("control epoch","23","71"),("service posture","standby","active"),("refresh window","11 seconds","31 seconds"),("replica policy","single","paired"),("telemetry tier","basic","extended"),("routing lane","auxiliary","primary"),("snapshot period","12 minutes","38 minutes")],
 "dev":[("worker quota","6","14"),("sampling regime","sparse","dense"),("archive tier","cold","warm"),("sync mode","deferred","immediate")],
 "test":[("dispatch plan","serial","parallel"),("retention window","18 hours","42 hours"),("failover mode","passive","automatic"),("checkpoint cadence","9 minutes","27 minutes")],
}

SPECS={
"corrects_current":("corrects","source",{
"train":["After the correction, what {attribute} should {subject} use?","Which {attribute} is valid once the error is fixed for {subject}?","What corrected {attribute} applies to {subject}?"],
"dev":["After rectification, which {attribute} remains authoritative for {subject}?","What {attribute} governs {subject} once the mistake is amended?"],
"test":["Once the erroneous record is set right, what {attribute} is in force for {subject}?","Which {attribute} survives the remedial update for {subject}?"]}),
"corrects_previous":("corrects","target",{
"train":["Which {attribute} did the correction replace for {subject}?","What earlier {attribute} became invalid after the fix for {subject}?","Which mistaken {attribute} was corrected for {subject}?"],
"dev":["Which {attribute} was displaced by the rectification for {subject}?","What {attribute} ceased to be authoritative after amendment for {subject}?"],
"test":["Which {attribute} belonged to the erroneous record that was set right for {subject}?","What {attribute} was retired by the remedial update for {subject}?"]}),
"supersedes_current":("supersedes","source",{
"train":["After supersession, what {attribute} should {subject} use?","Which {attribute} belongs to the replacement state for {subject}?","What {attribute} takes precedence for {subject}?"],
"dev":["Which {attribute} remains operative after the newer state displaces the older one for {subject}?","What {attribute} has precedence after replacement for {subject}?"],
"test":["Which {attribute} is controlling once the successor state takes over for {subject}?","What {attribute} belongs to the state that prevails over its predecessor for {subject}?"]}),
"supersedes_previous":("supersedes","target",{
"train":["Which {attribute} belonged to the superseded state for {subject}?","What {attribute} was replaced by the newer state for {subject}?","Which previous {attribute} lost precedence for {subject}?"],
"dev":["What {attribute} belonged to the displaced state for {subject}?","Which {attribute} ceased to be operative when the newer state took over for {subject}?"],
"test":["What {attribute} was attached to the predecessor that no longer governs {subject}?","Which {attribute} belonged to the state that yielded precedence for {subject}?"]}),
"temporal_current":("temporal_successor","source",{
"train":["What {attribute} belongs to the later state for {subject}?","Which {attribute} comes next in time for {subject}?","After the temporal transition, what {attribute} applies to {subject}?"],
"dev":["Which {attribute} is carried by the succeeding state for {subject}?","What {attribute} appears after the recorded transition for {subject}?"],
"test":["Which {attribute} occupies the subsequent point in the sequence for {subject}?","What {attribute} belongs to the state immediately following the earlier one for {subject}?"]}),
"temporal_previous":("temporal_successor","target",{
"train":["What {attribute} belongs to the earlier state for {subject}?","Which {attribute} comes before the successor for {subject}?","Before the temporal transition, what {attribute} applied to {subject}?"],
"dev":["Which {attribute} is carried by the preceding state for {subject}?","What {attribute} appears before the recorded transition for {subject}?"],
"test":["Which {attribute} occupies the prior point in the sequence for {subject}?","What {attribute} belongs to the state immediately preceding the later one for {subject}?"]}),
"causes_cause":("causes","source",{
"train":["Which {attribute} caused the linked outcome for {subject}?","What {attribute} initiated the documented effect for {subject}?","Which {attribute} is the causal condition for {subject}?"],
"dev":["What {attribute} is upstream of the observed consequence for {subject}?","Which {attribute} generated the resulting state for {subject}?"],
"test":["What {attribute} served as the antecedent producing the outcome for {subject}?","Which {attribute} is the originating factor in the causal link for {subject}?"]}),
"causes_effect":("causes","target",{
"train":["Which {attribute} resulted from the linked cause for {subject}?","What {attribute} is the documented effect for {subject}?","Which {attribute} is the causal consequence for {subject}?"],
"dev":["What {attribute} is downstream of the initiating condition for {subject}?","Which {attribute} was generated by the causal state for {subject}?"],
"test":["What {attribute} is the consequent produced by the antecedent for {subject}?","Which {attribute} is the resulting factor in the causal link for {subject}?"]}),
"supports_supported":("supports","target",{
"train":["Which {attribute} is supported by the linked evidence for {subject}?","What {attribute} is the claim receiving support for {subject}?","Which {attribute} is corroborated for {subject}?"],
"dev":["What {attribute} is backed by the other record for {subject}?","Which {attribute} is the evidentially sustained claim for {subject}?"],
"test":["What {attribute} is the conclusion warranted by the supporting record for {subject}?","Which {attribute} is the proposition strengthened by the evidence link for {subject}?"]}),
"supports_supporter":("supports","source",{
"train":["Which {attribute} belongs to the record providing support for {subject}?","What {attribute} comes from the supporting record for {subject}?","Which {attribute} is on the evidence-provider side for {subject}?"],
"dev":["What {attribute} belongs to the record that backs the claim for {subject}?","Which {attribute} is carried by the corroborating record for {subject}?"],
"test":["What {attribute} belongs to the evidentiary record that lends support for {subject}?","Which {attribute} is on the warranting side of the support link for {subject}?"]}),
"derived_item":("derived_from","source",{
"train":["Which {attribute} belongs to the derived record for {subject}?","What {attribute} was produced from the linked basis for {subject}?","Which {attribute} is the derivation result for {subject}?"],
"dev":["What {attribute} belongs to the record obtained from the other one for {subject}?","Which {attribute} is the dependent product of the derivation for {subject}?"],
"test":["What {attribute} belongs to the artifact yielded from its basis for {subject}?","Which {attribute} is the consequent item of the derivation relation for {subject}?"]}),
"derived_basis":("derived_from","target",{
"train":["Which {attribute} is the basis from which the other record was derived for {subject}?","What {attribute} underlies the derived record for {subject}?","Which {attribute} is the derivation basis for {subject}?"],
"dev":["What {attribute} belongs to the record the derived item depends on for {subject}?","Which {attribute} is the antecedent material of the derivation for {subject}?"],
"test":["What {attribute} belongs to the source basis that yielded the derived item for {subject}?","Which {attribute} is the underlying item on which the derivation rests for {subject}?"]}),
}

COUNTS={"train":30,"dev":8,"test":8}
ROLE_ID={"source":0,"target":1}

def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def field(name,text):return {"name":name,"text":text,"field_type_id":1,"provenance_id":2,"relation_role_id":1,"temporal_scope_id":1,"confidence":0.97,"missing":False}
def edge(s,t,r):return {"source":s,"target":t,"relation":r,"relation_type_id":relation_type_id(r),"confidence":1.0}

def make_pair(family,relation,role,split,index,offset):
    subjects=SUBJECTS[split]; attrs=ATTRIBUTES[split]; templates=SPECS[family][2][split]
    subject=subjects[(index+3*offset)%len(subjects)]
    attribute,a,b=attrs[(index+offset)%len(attrs)]
    query=templates[index%len(templates)].format(subject=subject,attribute=attribute)
    fields=[field("record_left",f"One record for {subject} reports {a} for {attribute}."),
            field("record_right",f"Another record for {subject} reports {b} for {attribute}.")]
    pair_id=f"N0V02-QRR-{family.upper()}-{split.upper()}-{index+1:02d}"
    def row(v,s,t):
        ans=s if role=="source" else t
        dist=[0.0,0.0];dist[ans]=1.0
        vals=[a,b]
        return {"id":f"{pair_id}-{v}","pair_id":pair_id,"variant":v,"family":family,"relation":relation,
                "semantic_role":role,"semantic_role_id":ROLE_ID[role],"split":split,"query_text":query,
                "fields":fields,"relations":[edge(s,t,relation)],"target_evidence_distribution":dist,
                "target_summary_text":f"The relation-semantic answer for {subject}'s {attribute} is {vals[ans]}.",
                "relation_essential":True,"paired_non_relation_inputs_identical":True,
                "target_flips_only_with_relation_direction":True,"query_names_endpoint_role_explicitly":False,
                "data_origin":"deterministic_public_synthetic_query_relation_role_router_v0_3",
                "generated_text":True,"identity_authority":False,"private_identity_content":False,
                "training_authorized":split=="train","frozen_latent_challenge_row_reused":False,
                "missing_evidence_localization_row_reused":False,"relation_semantic_v0_1_row_reused":False,
                "role_residual_v0_2_row_reused":False,"endpoint_repair_v0_2_heldout_row_reused":False}
    return [row("A",1,0),row("B",0,1)]

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",required=True);ap.add_argument("--manifest",required=True);a=ap.parse_args()
    rows=[]
    for off,(fam,(rel,role,_)) in enumerate(SPECS.items()):
        for split,count in COUNTS.items():
            for i in range(count):rows.extend(make_pair(fam,rel,role,split,i,off))
    grouped=defaultdict(list)
    for r in rows:grouped[r["pair_id"]].append(r)
    for pid,m in grouped.items():
        if len(m)!=2:raise SystemExit(f"pair size drift {pid}")
        x,y=sorted(m,key=lambda z:z["variant"])
        for k in ("fields","query_text","family","relation","semantic_role","semantic_role_id","split"):
            if x[k]!=y[k]:raise SystemExit(f"non-relation drift {pid}:{k}")
        ea,eb=x["relations"][0],y["relations"][0]
        if ea["source"]!=eb["target"] or ea["target"]!=eb["source"]:raise SystemExit(f"edge flip drift {pid}")
        if x["target_evidence_distribution"]==y["target_evidence_distribution"]:raise SystemExit(f"target flip drift {pid}")
    # Split template banks must be pairwise disjoint inside each family.
    for fam,(_rel,_role,banks) in SPECS.items():
        seen=set()
        for split in ("train","dev","test"):
            cur=set(banks[split])
            if seen & cur:raise SystemExit(f"template leakage {fam}:{split}")
            seen|=cur
    out=Path(a.output).resolve();man=Path(a.manifest).resolve()
    if out.exists() or man.exists():raise SystemExit("refusing overwrite")
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in rows),encoding="utf-8")
    manifest={"schema":"alice.eipm.n0.v02-query-relation-role-router-curriculum.v0.3",
      "status":"TRAIN_DEV_TEST_COMPILED_TEST_UNTOUCHED","rows":len(rows),"families":list(SPECS),
      "family_count":len(SPECS),"pairs_per_family":sum(COUNTS.values()),"split_pairs_per_family":COUNTS,
      "compiled_sha256":sha(out),"data_origin":"deterministic_public_synthetic_query_relation_role_router_v0_3",
      "train_dev_test_templates_lexically_disjoint":True,"subject_pools_split_disjoint":True,
      "attribute_value_pools_split_disjoint":True,"direct_semantic_role_supervision":True,
      "semantic_role_classes":["source","target"],"router_runtime_classes":["source","target","defer_to_parent"],
      "paired_non_relation_inputs_identical":True,"target_flips_only_with_relation_direction":True,
      "query_names_endpoint_role_explicitly":False,"frozen_latent_challenge_rows_used_for_training":False,
      "missing_evidence_localization_rows_used_for_training":False,"relation_semantic_v0_1_rows_reused":False,
      "role_residual_v0_2_rows_reused":False,"endpoint_repair_v0_2_heldout_rows_reused":False,
      "test_split_training_authorized":False,"dev_split_training_authorized":False,
      "train_split_training_authorized":True,"hard_parameter_ceiling":None}
    man.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(manifest,indent=2,sort_keys=True))
if __name__=="__main__":main()
