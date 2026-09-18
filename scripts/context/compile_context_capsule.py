#!/usr/bin/env python3
"""Compile a compact, source-pointer-first A.L.I.C.E. continuation capsule."""

from __future__ import annotations

import json
import re
from pathlib import Path

BASE=Path("research/graphify-context")
RESULT=BASE/"QUERY_RESULT.json"
OUT_JSON=BASE/"CONTEXT_CAPSULE.json"
OUT_MD=BASE/"CONTEXT_CAPSULE.md"


def main() -> None:
    data=json.loads(RESULT.read_text(encoding="utf-8"))
    cat=data.get("catalog_result") or {}
    active=cat.get("active_n0_state") or {}
    hits=(cat.get("hits") or [])[:12]

    graph_nodes=[]
    seen=set()
    for line in str(data.get("graphify_result_text","")).splitlines():
        if not line.startswith("NODE "):
            continue
        m=re.match(r"NODE (.*?) \[src=(.*?) loc=(.*?) community=(.*?)\]$",line)
        if not m:
            continue
        label,src,loc,community=m.groups()
        key=(label,src,loc)
        if key in seen:
            continue
        seen.add(key)
        graph_nodes.append({
            "label":label,
            "source":src,
            "location":loc,
            "community":community,
        })
        if len(graph_nodes)>=24:
            break

    capsule={
        "schema":"alice-sol-context-capsule-v1",
        "request_id":data.get("request_id"),
        "question":data.get("question"),
        "generated_at_utc":data.get("generated_at_utc"),
        "source_freshness":{
            "source_branch":data.get("graph_build",{}).get("source_work_branch"),
            "source_commit":data.get("graph_build",{}).get("source_work_commit"),
            "graph_extraction_tree":data.get("graph_build",{}).get("extraction_tree"),
            "catalog_source_commit":cat.get("catalog_status",{}).get("source_commit"),
        },
        "active_mission":active,
        "authoritative_source_pointers":hits,
        "graphify_navigation_hints":graph_nodes,
        "private_external_route_recommended":cat.get("private_external_route_recommended"),
        "private_external_route_reason":cat.get("private_external_route_reason"),
        "trust_contract":[
            "This capsule routes retrieval; it is not canonical A.L.I.C.E. truth.",
            "Open consequential claims in the original branch/path source before acting.",
            "Keep branch-specific, experimental, superseded and canonical states distinct.",
            "Graphify nodes are navigation hints only.",
            "If private external routing is recommended, search the private Library manifest/source corpus before declaring context complete.",
        ],
    }
    OUT_JSON.write_text(json.dumps(capsule,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")

    lines=[
        "# A.L.I.C.E. Sol Context Capsule",
        "",
        f"- Request: {capsule['request_id']}",
        f"- Question: {capsule['question']}",
        f"- Source: {capsule['source_freshness']['source_branch']} @ {capsule['source_freshness']['source_commit']}",
        f"- Catalog source: {capsule['source_freshness']['catalog_source_commit']}",
        "",
        "## Active mission",
        "",
        f"- Status: {active.get('status')}",
        f"- Source: {active.get('source_path')}",
        f"- N0 complete: {active.get('n0_complete')}",
        f"- Pass next action: {active.get('pass_next_action')}",
        f"- Fail next action: {active.get('fail_next_action')}",
        "",
        "## Source pointers",
        "",
    ]
    for h in hits:
        lines.append(
            f"- [{h.get('score')}] {h.get('branch')}:{h.get('path')} — {h.get('title')}"
        )
        declared=h.get("declared") or {}
        if declared.get("status"):
            lines.append(f"  - status: {declared['status']}")
        if declared.get("supersession"):
            lines.append(f"  - supersession: {declared['supersession']}")
    lines.extend(["","## Graphify navigation hints",""])
    for n in graph_nodes[:16]:
        lines.append(f"- {n['label']} -> {n['source']}:{n['location']}")
    lines.extend([
        "",
        "## External/private routing",
        "",
        f"- Recommended: {capsule['private_external_route_recommended']}",
        f"- Reason: {capsule['private_external_route_reason']}",
        "",
        "## Trust contract",
        "",
    ])
    for item in capsule["trust_contract"]:
        lines.append(f"- {item}")
    OUT_MD.write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
