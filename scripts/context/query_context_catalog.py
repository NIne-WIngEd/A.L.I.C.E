#!/usr/bin/env python3
"""Query A.L.I.C.E. routing catalogs with a small deterministic scorer.

This is not semantic truth inference. It ranks branch-qualified source pointers
so the agent can open original authority with minimal broad reading.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path("research/graphify-context/catalog")

STOP = {
    "the","and","for","with","from","that","this","what","where","which","when",
    "does","did","have","has","had","into","about","alice","a.l.i.c.e","current",
    "now","our","are","was","were","how","why","who","can","could","should",
}


def tokens(text: str) -> list[str]:
    raw = re.findall(r"[a-z0-9][a-z0-9_.+-]{2,}", text.lower())
    return [x for x in raw if x not in STOP]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows=[]
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def haystack(row: dict) -> tuple[str,str,str]:
    title=str(row.get("title",""))
    path=str(row.get("path",""))
    extra=json.dumps({
        "headings": row.get("headings",[]),
        "declared": row.get("declared",{}),
        "matched_terms": row.get("matched_terms",[]),
        "supersession_hints": row.get("supersession_hints",[]),
    },ensure_ascii=False)
    return title.lower(), path.lower(), extra.lower()


def score(row: dict, q: str, qtokens: list[str], catalog: str) -> int:
    title,path,extra=haystack(row)
    s=0
    phrase=q.lower().strip()
    if phrase and phrase in title:
        s+=24
    if phrase and phrase in path:
        s+=20
    for t in qtokens:
        if t in title:
            s+=7
        if t in path:
            s+=6
        if t in extra:
            s+=2
    branch=row.get("branch")
    if branch=="alice-eipm-v1-build":
        s+=3
    if row.get("branch_scope")=="branch_specific":
        s+=1
    declared=row.get("declared") or {}
    if declared.get("supersession") and any(x in qtokens for x in ("supersede","superseded","obsolete","replace","replaced")):
        s+=8
    if catalog=="failure" and any(x in qtokens for x in ("fail","failed","failure","error","repair","magnolia","kaggle","powershell","udocker")):
        s+=8
    if catalog=="frontier" and any(x in qtokens for x in ("frontier","research","competitor","architecture","model","eipm","personality")):
        s+=8
    if catalog=="supersession" and any(x in qtokens for x in ("supersede","superseded","obsolete","old","replace","replaced","deprecated")):
        s+=8
    return s


def compact(row: dict, catalog: str, s: int) -> dict:
    return {
        "score": s,
        "catalog": catalog,
        "branch": row.get("branch"),
        "branch_scope": row.get("branch_scope"),
        "path": row.get("path"),
        "blob_sha": row.get("blob_sha"),
        "category": row.get("category"),
        "title": row.get("title"),
        "declared": row.get("declared",{}),
        "matched_terms": row.get("matched_terms",[]),
        "supersession_hints": row.get("supersession_hints",[])[:4],
    }


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--top",type=int,default=18)
    ap.add_argument("--output",default="research/graphify-context/CATALOG_QUERY_RESULT.json")
    args=ap.parse_args()

    q=args.question.strip()
    if not q:
        raise SystemExit("question required")
    qtok=tokens(q)

    catalogs={
        "document": load_jsonl(ROOT/"DOCUMENT_CATALOG.jsonl"),
        "failure": load_jsonl(ROOT/"FAILURE_LESSON_CATALOG.jsonl"),
        "frontier": load_jsonl(ROOT/"FRONTIER_RESEARCH_CATALOG.jsonl"),
        "supersession": load_jsonl(ROOT/"SUPERSESSION_CATALOG.jsonl"),
    }

    candidates=[]
    for name,rows in catalogs.items():
        for row in rows:
            s=score(row,q,qtok,name)
            if s>0:
                candidates.append(compact(row,name,s))

    # Deduplicate the same branch/path surfaced from several catalogs, retaining
    # the best score and noting the other matching surfaces.
    merged={}
    for hit in sorted(candidates,key=lambda x:(-x["score"],x.get("branch") or "",x.get("path") or "")):
        key=(hit.get("branch"),hit.get("path"),hit.get("blob_sha"))
        if key not in merged:
            hit["catalogs"]=[hit.pop("catalog")]
            merged[key]=hit
        else:
            name=hit["catalog"]
            if name not in merged[key]["catalogs"]:
                merged[key]["catalogs"].append(name)
            merged[key]["score"]=max(merged[key]["score"],hit["score"])

    ranked=sorted(merged.values(),key=lambda x:(-x["score"],x.get("branch") or "",x.get("path") or ""))[:max(1,min(args.top,50))]

    active={}
    p=ROOT/"ACTIVE_N0_STATE.json"
    if p.exists():
        active=json.loads(p.read_text(encoding="utf-8"))

    catalog_status=json.loads((ROOT/"CATALOG_STATUS.json").read_text(encoding="utf-8"))

    qlow=q.lower()
    private_terms=("comic","old chat","previous chat","prior chat","conversation","pdf","private","elaina source","handoff export")
    external_recommended=any(t in qlow for t in private_terms)
    low_confidence=(not ranked) or ranked[0]["score"]<10

    result={
        "schema":"alice-context-catalog-query-v1",
        "question":q,
        "query_tokens":qtok,
        "catalog_status":catalog_status,
        "active_n0_state":active,
        "hits":ranked,
        "private_external_route_recommended": external_recommended or low_confidence,
        "private_external_route_reason": (
            "question references a private/external source class"
            if external_recommended else
            "public routing catalogs have low confidence"
            if low_confidence else
            None
        ),
        "authority":"routing-only",
    }
    Path(args.output).write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
