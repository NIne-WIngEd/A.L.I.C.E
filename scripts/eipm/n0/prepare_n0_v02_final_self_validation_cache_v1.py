from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from prepare_n0_v02_qsre_production_cache_v1 import (
    build_split,
    encode_texts,
    read_jsonl,
)
from qsre_production_runtime import (
    load_frozen_semantic_model,
    load_tokenizer,
    sha256,
)


CACHE_SCHEMA = "alice.eipm.n0.qsre-final-self-validation-cache.v1"


@torch.inference_mode()
def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--relational-corpus",required=True)
    p.add_argument("--manifest",required=True)
    p.add_argument("--schema-cache",required=True)
    p.add_argument("--semantic-config",required=True)
    p.add_argument("--semantic-checkpoint",required=True)
    p.add_argument("--tokenizer-dir",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--receipt",required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--batch-size",type=int,default=16)
    args=p.parse_args()

    corpus_path=Path(args.relational_corpus)
    manifest_path=Path(args.manifest)
    schema_cache_path=Path(args.schema_cache)
    semantic_checkpoint=Path(args.semantic_checkpoint)
    output_path=Path(args.output)
    receipt_path=Path(args.receipt)
    device=torch.device(args.device)

    rows=read_jsonl(corpus_path)
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status")!="FROZEN_BY_BUILD_SOURCE_BEFORE_PRODUCTION_MODEL_RESULTS":
        raise SystemExit("final self-validation manifest is not frozen")
    if manifest.get("relational_sha256")!=sha256(corpus_path):
        raise SystemExit("final relational corpus hash drift")
    if manifest.get("training_authorized") is not False:
        raise SystemExit("final validation corpus unexpectedly authorizes training")
    if len(rows)!=int(manifest["relational_rows"]):
        raise SystemExit("final relational row count drift")
    if any(row.get("private_identity_data") is not False for row in rows):
        raise SystemExit("private identity data entered final relational validation cache")

    schema_payload=torch.load(schema_cache_path,map_location="cpu")
    if schema_payload.get("schema")!="alice.eipm.n0.qsre-production-schema-cache.v1":
        raise SystemExit("final dynamic schema cache drift")
    relation_key_to_index={
        str(key):int(value)
        for key,value in schema_payload["relation_key_to_index"].items()
    }
    type_to_index={
        str(key):int(value)
        for key,value in schema_payload["type_to_index"].items()
    }
    final_only=set(str(value) for value in manifest["final_only_relation_keys"])
    if not final_only.issubset(relation_key_to_index):
        raise SystemExit("final-only relation missing from frozen final schema")

    model=load_frozen_semantic_model(
        semantic_config_path=Path(args.semantic_config),
        semantic_checkpoint=semantic_checkpoint,
        device=device,
    )
    tokenizer=load_tokenizer(Path(args.tokenizer_dir))

    query_views=[list(row["query_views"]) for row in rows]
    if any(len(views)!=2 for views in query_views):
        raise SystemExit("final relational query-view count drift")
    query_texts=[text for views in query_views for text in views]
    field_texts=[
        field["text"]
        for row in rows
        for field in row["fields"]
    ]
    query_hidden_flat,query_mask_flat=encode_texts(
        texts=query_texts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=128,
        batch_size=args.batch_size,
        all_hidden_states=True,
    )
    query_hidden=query_hidden_flat.reshape(
        len(rows),2,
        query_hidden_flat.size(1),
        query_hidden_flat.size(2),
        query_hidden_flat.size(3),
    )
    query_mask=query_mask_flat.reshape(
        len(rows),2,query_mask_flat.size(1),
    )
    field_hidden,field_mask=encode_texts(
        texts=field_texts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=96,
        batch_size=args.batch_size,
        all_hidden_states=False,
    )

    split=build_split(
        rows,
        query_hidden=query_hidden,
        query_mask=query_mask,
        field_token=field_hidden,
        field_token_mask_flat=field_mask,
        relation_key_to_index=relation_key_to_index,
        type_to_index=type_to_index,
    )
    split["final_only_relation"]=torch.tensor(
        [bool(row["final_only_relation"]) for row in rows],
        dtype=torch.bool,
    )
    split["relational_required_for_integrated_answer"]=torch.tensor(
        [bool(row["relational_required_for_integrated_answer"]) for row in rows],
        dtype=torch.bool,
    )
    split["query_texts"]=[str(row["query_text"]) for row in rows]
    split["target_summary_texts"]=[str(row["target_summary_text"]) for row in rows]

    payload={
        "schema":CACHE_SCHEMA,
        "relational_corpus_sha256":sha256(corpus_path),
        "manifest_sha256":sha256(manifest_path),
        "schema_cache_sha256":sha256(schema_cache_path),
        "semantic_checkpoint_sha256":sha256(semantic_checkpoint),
        "rows":split,
        "private_identity_data":False,
        "training_authorized":False,
        "gradient":False,
    }
    output_path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(payload,output_path)
    receipt={
        "schema":"alice.eipm.n0.qsre-final-self-validation-cache-result.v1",
        "status":"PASS_QSRE_FINAL_SELF_VALIDATION_CACHE_PREPARATION",
        "relational_corpus_sha256":payload["relational_corpus_sha256"],
        "manifest_sha256":payload["manifest_sha256"],
        "schema_cache_sha256":payload["schema_cache_sha256"],
        "semantic_checkpoint_sha256":payload["semantic_checkpoint_sha256"],
        "prepared_cache_sha256":sha256(output_path),
        "rows":len(rows),
        "query_views_per_row":2,
        "query_hidden_depth":int(query_hidden.size(2)),
        "semantic_dim":int(query_hidden.size(-1)),
        "field_token_width":int(field_hidden.size(1)),
        "final_only_rows":int(split["final_only_relation"].sum().item()),
        "relational_required_rows":int(
            split["relational_required_for_integrated_answer"].sum().item()
        ),
        "training_authorized":False,
        "optimizer":False,
        "gradient":False,
        "private_identity_data":False,
    }
    receipt_path.parent.mkdir(parents=True,exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
