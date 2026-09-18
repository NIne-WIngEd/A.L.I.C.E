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
    active=cat.get("active_mission_state") or cat.get("active_n0_state") or {}
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

    stable=(active.get("stable_build_base") or {})
    frontier=(active.get("unmerged_experiment_frontier") or {})
    freshness=(active.get("continuity_freshness") or {})
    maximal_heads=frontier.get("maximal_heads") or []

    capsule={
        "schema":"alice-sol-context-capsule-v2",
        "request_id":data.get("request_id"),
        "question":data.get("question"),
        "generated_at_utc":data.get("generated_at_utc"),
        "source_freshness":{
            "stable_build_branch":stable.get("branch") or cat.get("catalog_status",{}).get("source_branch"),
            "stable_build_commit":stable.get("commit") or cat.get("catalog_status",{}).get("source_commit"),
            "catalog_source_commit":cat.get("catalog_status",{}).get("source_commit"),
            "graphify_used":bool(data.get("graphify_used")),
            "graph_source_branch":data.get("graph_build",{}).get("source_work_branch") if data.get("graphify_used") else None,
            "graph_source_commit":data.get("graph_build",{}).get("source_work_commit") if data.get("graphify_used") else None,
            "graph_extraction_tree":data.get("graph_build",{}).get("extraction_tree") if data.get("graphify_used") else None,
            "continuity_branch_head":active.get("continuity_branch_head"),
            "continuity_stale_relative_to_experiment_frontier":freshness.get("stale_relative_to_unmerged_experiment_frontier"),
            "maximal_unmerged_experiment_heads":[
                {
                    "branch":row.get("branch"),
                    "head":row.get("head"),
                    "head_committed_at":row.get("head_committed_at"),
                    "head_message":row.get("head_message"),
                }
                for row in maximal_heads
            ],
        },
        "routing_decision":data.get("routing_decision"),
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
            "If continuity is stale relative to unmerged experiment heads, inspect those heads before execution.",
            "If private external routing is recommended, search the private Library manifest/source corpus before declaring context complete.",
        ],
    }
    OUT_JSON.write_text(json.dumps(capsule,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")

    lines=[
        "# A.L.I.C.E. Sol Context Capsule",
        "",
        f"- Request: {capsule['request_id']}",
        f"- Question: {capsule['question']}",
        f"- Stable build: {capsule['source_freshness']['stable_build_branch']} @ {capsule['source_freshness']['stable_build_commit']}",
        f"- Catalog source: {capsule['source_freshness']['catalog_source_commit']}",
        f"- Graphify used: {capsule['source_freshness']['graphify_used']}",
        "",
        "## Active mission",
        "",
        f"- Mission schema: {active.get('schema')}",
        f"- Implementation status: {(active.get('implementation_state') or active).get('status')}",
        f"- Implementation source: {(active.get('implementation_state') or active).get('source_path')}",
        f"- Continuity overlay: {(active.get('continuity_overlay') or {}).get('path')}",
        f"- Latest observed Magnolia job: {(active.get('continuity_overlay') or {}).get('magnolia_job_observed')}",
        f"- Continuity stale vs experiment frontier: {freshness.get('stale_relative_to_unmerged_experiment_frontier')}",
        f"- Execution rule: {active.get('execution_rule')}",
        "",
        "## Unmerged experiment frontier",
        "",
    ]
    if maximal_heads:
        for row in maximal_heads:
            lines.append(
                f"- {row.get('branch')} @ {row.get('head')} — {row.get('head_message')}"
            )
    else:
        lines.append("- none")
    lines.extend([
        "",
        "## Source pointers",
        "",
    ])
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
    if data.get("graphify_used"):
        for n in graph_nodes[:16]:
            lines.append(f"- {n['label']} -> {n['source']}:{n['location']}")
    else:
        lines.append("- skipped: document/branch/private routing was sufficient")
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
