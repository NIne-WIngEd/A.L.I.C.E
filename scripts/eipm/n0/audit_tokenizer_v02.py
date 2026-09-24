#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
from alice_personality.n0.source_authority_v1 import require_clean_exact_revision


SPECIAL_TOKEN_IDS={"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"[MASK]":4}
PASS="PASS_N0_TOKENIZER_STRESS_V1"
SCHEMA="alice.eipm.n0.tokenizer-stress-audit.v1"
REQUIRED_STRESS_FAMILIES=(
    "byte_fallback_oov",
    "unicode_normalization",
    "heldout_relation_factor_fragmentation",
    "long_entity",
    "punctuation_code_math",
    "multilingual",
)
fragmentation_limits={
    "heldout_relation_factor_max_tokens_per_word":8.0,
    "heldout_relation_factor_max_tokens_per_character":1.0,
}


def sha256_file(path: str | Path) -> str:
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_revision(value: str) -> str:
    revision=str(value).strip().lower()
    if len(revision)!=40 or any(ch not in "0123456789abcdef" for ch in revision):
        raise SystemExit("source revision must be exact 40-hex git commit")
    return revision


def stress_cases() -> dict[str,list[str]]:
    # These are fixed before candidate training. They exercise tokenizer
    # behavior only; they are not semantic-model labels or FINAL examples.
    return {
        "byte_fallback_oov":[
            "Rare symbols: 🧬 🛰️ 🜁 𐍈 ∰ ⧉ ⟂ ⟹",
            "Mixed bytes-as-Unicode: café naïve jalapeño façade Ångström",
        ],
        "unicode_normalization":[
            "Café Ångström résumé coöperate",
            unicodedata.normalize("NFD","Café Ångström résumé coöperate"),
        ],
        "heldout_relation_factor_fragmentation":[
            "the organization that the person was appointed to lead",
            "the location in which the event was formally established",
            "reverse traversal from the stated object toward the originating subject",
            "prefer evidence with stronger temporal consistency and provenance reliability",
            "defer when the relation is unsupported by the available evidence",
            "aggregate multiple compatible support paths without treating availability as relevance",
        ],
        "long_entity":[
            "International_Consortium_for_Cross_Domain_Semantic_Reasoning_"
            + "_".join(f"Division{i:03d}" for i in range(256)),
        ],
        "punctuation_code_math":[
            r"f(x)=Σ_{i=1}^{n} w_i·x_i; if (score >= 0.95) { return A->B::C; } // λ, ∂, ≈, !=, <=, &&",
            r"path=/alpha/beta?q=a%20b&x=1#frag; regex=^[A-Z][a-z]+\d{2,4}$; JSON={\"k\":[1,2,3]}",
        ],
        "multilingual":[
            "বাংলা ভাষায় সম্পর্ক ও প্রমাণের বর্ণনা",
            "日本語で関係と証拠を説明する文章",
            "العلاقة بين الكيانين مدعومة بالدليل",
            "Связь между сущностями подтверждается доказательством",
            "La relación entre las entidades está respaldada por evidencia",
            "La relation entre les entités est étayée par des preuves",
        ],
    }


def encode_exact(tokenizer: Any, text: str) -> tuple[list[int],str]:
    encoded=tokenizer.encode(text,add_special_tokens=False)
    ids=list(encoded.ids)
    if not ids:
        raise ValueError("zero-token encoding")
    if any(token_id==SPECIAL_TOKEN_IDS["[UNK]"] for token_id in ids):
        raise ValueError("[UNK] emitted")
    decoded=tokenizer.decode(ids,skip_special_tokens=True)
    if unicodedata.normalize("NFC",decoded)!=unicodedata.normalize("NFC",text):
        raise ValueError("normalized round-trip mismatch")
    return ids,decoded


def main() -> None:
    parser=argparse.ArgumentParser(
        description="Audit the governed Alice N0 v0.2.1 tokenizer and exact P40 stress surface."
    )
    parser.add_argument("--corpus-dir",required=True)
    parser.add_argument("--tokenizer-dir",required=True)
    parser.add_argument("--source-revision",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--train-rows-per-source",type=int,default=64)
    args=parser.parse_args()

    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise SystemExit("Install requirements-n0.txt before auditing the tokenizer") from exc

    source_revision=exact_revision(args.source_revision)
    require_clean_exact_revision(
        expected_revision=source_revision,label="P40 tokenizer stress audit"
    )
    if int(args.train_rows_per_source)<=0:
        raise SystemExit("train-rows-per-source must be positive")

    corpus_dir=Path(args.corpus_dir).resolve()
    tokenizer_dir=Path(args.tokenizer_dir).resolve()
    tokenizer_path=tokenizer_dir/"tokenizer.json"
    tokenizer_receipt_path=tokenizer_dir/"tokenizer_receipt.json"
    corpus_receipt_path=corpus_dir/"corpus_receipt.json"
    for path in (tokenizer_path,tokenizer_receipt_path,corpus_receipt_path):
        if not path.is_file():
            raise SystemExit(f"required audit artifact missing: {path}")

    receipt=json.loads(tokenizer_receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema")!="alice.eipm.n0.tokenizer-receipt.v0.2":
        raise SystemExit("unexpected tokenizer receipt schema")
    if receipt.get("tokenizer_sha256")!=sha256_file(tokenizer_path):
        raise SystemExit("tokenizer hash does not match tokenizer receipt")
    if receipt.get("corpus_receipt_sha256")!=sha256_file(corpus_receipt_path):
        raise SystemExit("tokenizer receipt is not bound to this tokenizer corpus")
    for key in ("private_identity_data","private_identity_gradient","model_training_performed"):
        if receipt.get(key) is not False:
            raise SystemExit(f"tokenizer receipt must declare {key}=false")

    tokenizer=Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size()!=48_000:
        raise SystemExit(f"expected 48000-token vocabulary, got {tokenizer.get_vocab_size()}")
    for token,expected_id in SPECIAL_TOKEN_IDS.items():
        actual=tokenizer.token_to_id(token)
        if actual!=expected_id:
            raise SystemExit(f"special token {token} expected id {expected_id}, got {actual}")

    corpus_receipt=json.loads(corpus_receipt_path.read_text(encoding="utf-8"))
    shard_paths:list[Path]=[]
    for source in corpus_receipt.get("sources",[]):
        for shard in source.get("shards",[]):
            path=corpus_dir/str(shard["path"])
            if not path.is_file() or sha256_file(path)!=str(shard["sha256"]):
                raise SystemExit(f"corpus shard failed audit hash verification: {path}")
            shard_paths.append(path)
    if not shard_paths:
        raise SystemExit("tokenizer stress audit found no governed corpus shards")

    train_kept:Counter[str]=Counter()
    split_rows:Counter[str]=Counter()
    split_chars:Counter[str]=Counter()
    split_tokens:Counter[str]=Counter()
    source_chars:Counter[str]=Counter()
    source_tokens:Counter[str]=Counter()
    unknown_tokens=0
    roundtrip_failures:list[dict[str,Any]]=[]

    for path in sorted(shard_paths):
        with path.open("r",encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row=json.loads(line)
                source_id=str(row.get("source_id") or "unknown")
                split=str(row.get("split") or "unknown")
                if split=="train":
                    if train_kept[source_id]>=args.train_rows_per_source:
                        continue
                    train_kept[source_id]+=1
                text=unicodedata.normalize("NFC",str(row.get("text") or ""))
                if not text:
                    continue
                encoded=tokenizer.encode(text,add_special_tokens=False)
                token_count=len(encoded.ids)
                if token_count==0:
                    raise SystemExit(
                        f"tokenizer produced zero tokens for non-empty row from {source_id}"
                    )
                unknown_tokens+=sum(
                    token_id==SPECIAL_TOKEN_IDS["[UNK]"] for token_id in encoded.ids
                )
                decoded=tokenizer.decode(encoded.ids,skip_special_tokens=True)
                if unicodedata.normalize("NFC",decoded)!=text and len(roundtrip_failures)<10:
                    roundtrip_failures.append({
                        "source_id":source_id,
                        "split":split,
                        "text_sha256":row.get("text_sha256"),
                    })
                split_rows[split]+=1
                split_chars[split]+=len(text)
                split_tokens[split]+=token_count
                source_chars[source_id]+=len(text)
                source_tokens[source_id]+=token_count

    if unknown_tokens:
        raise SystemExit(f"tokenizer emitted {unknown_tokens} [UNK] tokens during corpus audit")
    if roundtrip_failures:
        raise SystemExit(
            f"tokenizer round-trip failed on sampled corpus rows: {roundtrip_failures[:3]}"
        )
    if split_rows["dev"]<1 or split_rows["test"]<1:
        raise SystemExit(
            f"audit requires held-out coverage; observed dev={split_rows['dev']} "
            f"test={split_rows['test']}"
        )

    cases=stress_cases()
    if set(cases)!=set(REQUIRED_STRESS_FAMILIES):
        raise SystemExit("tokenizer stress family registry drift")

    stress_receipts:dict[str,list[dict[str,Any]]]={}
    errors:list[str]=[]
    for family in REQUIRED_STRESS_FAMILIES:
        family_receipts=[]
        for index,text in enumerate(cases[family]):
            try:
                ids,_=encode_exact(tokenizer,text)
                words=max(1,len(text.split()))
                chars=max(1,len(text))
                tokens=len(ids)
                row={
                    "case_index":index,
                    "chars":chars,
                    "words":words,
                    "tokens":tokens,
                    "tokens_per_word":tokens/words,
                    "tokens_per_character":tokens/chars,
                    "normalized_sha256":hashlib.sha256(
                        unicodedata.normalize("NFC",text).encode("utf-8")
                    ).hexdigest(),
                }
                if family=="heldout_relation_factor_fragmentation":
                    if row["tokens_per_word"]>fragmentation_limits[
                        "heldout_relation_factor_max_tokens_per_word"
                    ]:
                        raise ValueError("heldout relation/factor token-per-word fragmentation limit exceeded")
                    if row["tokens_per_character"]>fragmentation_limits[
                        "heldout_relation_factor_max_tokens_per_character"
                    ]:
                        raise ValueError("heldout relation/factor token-per-character fragmentation limit exceeded")
                family_receipts.append(row)
            except Exception as exc:
                errors.append(f"{family}[{index}]: {type(exc).__name__}: {exc}")
        stress_receipts[family]=family_receipts

    family_coverage={
        family:len(stress_receipts.get(family,[]))
        for family in REQUIRED_STRESS_FAMILIES
    }
    if any(count!=len(cases[family]) for family,count in family_coverage.items()):
        errors.append("tokenizer stress family coverage incomplete")

    # Explicitly verify canonical-equivalent Unicode forms after NFC
    # normalization. Raw token IDs need not be identical if the tokenizer owns
    # no internal normalizer, but normalized text must remain lossless/OOV-free.
    unicode_cases=cases["unicode_normalization"]
    normalized_ids=[
        tokenizer.encode(unicodedata.normalize("NFC",text),add_special_tokens=False).ids
        for text in unicode_cases
    ]
    unicode_normalization_equivalent=all(
        list(ids)==list(normalized_ids[0]) for ids in normalized_ids[1:]
    )
    if not unicode_normalization_equivalent:
        errors.append("Unicode NFC-equivalent forms tokenize differently after normalization")

    def ratio(chars: int,tokens: int) -> float | None:
        return chars/tokens if tokens else None

    result={
        "schema":SCHEMA,
        "status":PASS if not errors else "FAIL_N0_TOKENIZER_STRESS_V1",
        "source_revision":source_revision,
        "auditor_sha256":sha256_file(Path(__file__).resolve()),
        "tokenizer_sha256":sha256_file(tokenizer_path),
        "tokenizer_receipt_sha256":sha256_file(tokenizer_receipt_path),
        "corpus_receipt_sha256":sha256_file(corpus_receipt_path),
        "vocab_size":tokenizer.get_vocab_size(),
        "special_token_ids":SPECIAL_TOKEN_IDS,
        "unknown_token_count":0,
        "roundtrip_failures":0,
        "sampled_rows_by_split":dict(sorted(split_rows.items())),
        "sampled_chars_by_split":dict(sorted(split_chars.items())),
        "sampled_tokens_by_split":dict(sorted(split_tokens.items())),
        "chars_per_token_by_split":{
            split:ratio(split_chars[split],split_tokens[split])
            for split in sorted(split_rows)
        },
        "chars_per_token_by_source":{
            source:ratio(source_chars[source],source_tokens[source])
            for source in sorted(source_chars)
        },
        "required_stress_families":list(REQUIRED_STRESS_FAMILIES),
        "stress_family_coverage":family_coverage,
        "stress_case_count":sum(family_coverage.values()),
        "stress_receipts":stress_receipts,
        "fragmentation_limits":fragmentation_limits,
        "unicode_normalization_equivalent":unicode_normalization_equivalent,
        "errors":errors,
        "private_identity_data":False,
        "private_identity_gradient":False,
        "model_training_performed":False,
        "gradient":False,
        "optimizer":False,
        "gpu_training_authorized":False,
        "final_results_observed":False,
        "final_opening_authorized":False,
        "n0_complete":False,
    }
    output=Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite tokenizer stress receipt: {output}")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,sort_keys=True))
    if errors:
        raise SystemExit(2)


if __name__=="__main__":
    main()
