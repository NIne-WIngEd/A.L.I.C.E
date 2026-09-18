#!/usr/bin/env python3
"""Build a deterministic, branch-qualified lexical index for A.L.I.C.E. context docs.

No model or external service is used. The index is routing-only and stores token ->
document-key postings so uncommon facts buried in document bodies can be located
without copying the full corpus into a prompt.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

CATALOG = Path("research/graphify-context/catalog")
DOCS = CATALOG / "DOCUMENT_CATALOG.jsonl"
OUT = CATALOG / "lexical"
MAX_DOC_BYTES = 1_500_000
MAX_TOKEN_LEN = 96
COMMON_FRACTION_CUTOFF = 0.35

STOP = {
    "the","and","for","with","from","that","this","what","where","which","when",
    "does","did","have","has","had","into","about","alice","a.l.i.c.e","current",
    "now","our","are","was","were","how","why","who","can","could","should",
    "would","will","than","then","them","they","their","there","these","those",
    "been","being","also","only","not","but","you","your","its","itself","use",
    "using","used","each","same","more","most","some","any","all","one","two",
    "new","old","per","via","may","must","shall","true","false","none","null",
    "status","schema","version","file","path","source","result","results",
}

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_.+:/-]{2,}", re.I)


def git_show(branch: str, path: str) -> str | None:
    p = subprocess.run(
        ["git", "show", f"refs/remotes/origin/{branch}:{path}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if p.returncode != 0 or len(p.stdout) > MAX_DOC_BYTES:
        return None
    return p.stdout.decode("utf-8", errors="replace")


def doc_key(row: dict) -> str:
    raw = "\0".join([
        str(row.get("branch","")),
        str(row.get("path","")),
        str(row.get("blob_sha","")),
    ]).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def normalize_tokens(text: str) -> set[str]:
    out=set()
    for raw in TOKEN_RE.findall(text.lower()):
        token=raw.strip("._:+/-")
        if not token or token in STOP:
            continue
        if len(token) < 3 or len(token) > MAX_TOKEN_LEN:
            continue
        if token.isdigit() and len(token) < 5:
            continue
        out.add(token)
    return out


def shard_for(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[0]


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",",":")) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    rows=[]
    for line in DOCS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("POSTINGS_*.json"):
        old.unlink()

    token_docs: dict[str,set[str]] = defaultdict(set)
    indexed_docs={}
    skipped=[]

    for row in rows:
        key=doc_key(row)
        text=git_show(str(row.get("branch")), str(row.get("path")))
        if text is None:
            skipped.append({
                "doc_key":key,
                "branch":row.get("branch"),
                "path":row.get("path"),
                "reason":"unreadable_or_over_size_limit",
            })
            continue

        # Include metadata too so exact technical names in titles/headings survive.
        meta="\n".join([
            str(row.get("title","")),
            str(row.get("path","")),
            " ".join(map(str,row.get("headings",[]))),
            json.dumps(row.get("declared",{}),ensure_ascii=False),
            text,
        ])
        toks=normalize_tokens(meta)
        indexed_docs[key]={
            "branch":row.get("branch"),
            "path":row.get("path"),
            "blob_sha":row.get("blob_sha"),
        }
        for token in toks:
            token_docs[token].add(key)

    n=max(1,len(indexed_docs))
    max_df=max(2,int(n*COMMON_FRACTION_CUTOFF))

    shards={h:{} for h in "0123456789abcdef"}
    dropped_common=[]
    for token,keys in token_docs.items():
        if len(keys)>max_df:
            dropped_common.append({"token":token,"document_frequency":len(keys)})
            continue
        shards[shard_for(token)][token]=sorted(keys)

    for h,postings in shards.items():
        write_json(
            OUT/f"POSTINGS_{h}.json",
            {
                "schema":"alice-context-lexical-postings-v1",
                "shard":h,
                "postings":dict(sorted(postings.items())),
            },
        )

    write_json(
        OUT/"DOC_LOOKUP.json",
        {
            "schema":"alice-context-lexical-doc-lookup-v1",
            "documents":indexed_docs,
        },
    )
    write_json(
        OUT/"LEXICAL_STATUS.json",
        {
            "schema":"alice-context-lexical-status-v1",
            "document_catalog_rows":len(rows),
            "indexed_documents":len(indexed_docs),
            "skipped_documents":len(skipped),
            "unique_tokens_before_common_filter":len(token_docs),
            "unique_tokens_indexed":sum(len(x) for x in shards.values()),
            "common_document_frequency_cutoff":max_df,
            "common_fraction_cutoff":COMMON_FRACTION_CUTOFF,
            "shard_count":16,
            "authority":"routing-only",
            "external_model_used":False,
            "skipped":skipped[:200],
            "dropped_common_terms":sorted(
                dropped_common,
                key=lambda x:(-x["document_frequency"],x["token"]),
            )[:200],
        },
    )


if __name__=="__main__":
    main()
