from cognitive_kernel import (
    AttentionDecision,
    AttentionRankEntry,
    ProductHostScope,
    ProvenanceReference,
    WorkspaceItemProjection,
    WorkspaceLayout,
    canonical_sha256,
)


def scope(product="alice", host="host-a"):
    return ProductHostScope.create(
        product_id=product,
        host_instance_id=host,
        schema_version="1.0.0",
        encryption_domain=f"private-{host}",
    )


def provenance():
    return ProvenanceReference.create(
        provenance_type="derived_inference",
        source_reference_ids=("source-1",),
        derivation_activity_id="derive-attention-1",
        responsible_component="synthetic-attention-policy",
        confidence=0.8,
    )


def digest(label):
    return canonical_sha256({"label": label})


def entry(
    *,
    key="entry-1",
    reference="node-1",
    rank=1,
    selected=True,
    product="alice",
    host="host-a",
    subject_type="mission_node",
    priority="host_engaged",
    host_override="none",
    protected_reason=None,
    suppression_reason=None,
    privacy="ordinary",
):
    del privacy
    return AttentionRankEntry.create(
        entry_key=key,
        scope=scope(product, host),
        reference_id=reference,
        subject_type=subject_type,
        mission_id=("mission-1" if subject_type in {"mission_node", "result_capsule"} else None),
        node_id=(reference if subject_type == "mission_node" else None),
        capsule_id=(reference if subject_type == "result_capsule" else None),
        observed_at=f"2026-08-01T10:{rank:02d}:00Z",
        state_digest=digest(f"state-{key}"),
        priority_class=priority,
        rank=rank,
        score=max(0.0, 1.0 - rank / 20),
        interruption_cost=min(1.0, rank / 20),
        protected_interrupt_reason=protected_reason,
        host_override=host_override,
        selected=selected,
        reason_codes=(priority, "synthetic_evidence"),
        suppression_reason=suppression_reason,
        policy_bindings=("attention-policy-v1",),
    )


def decision(entries, *, product="alice", host="host-a", limit=3):
    return AttentionDecision.create(
        decision_key="decision-1",
        scope=scope(product, host),
        decided_at="2026-08-01T10:10:00Z",
        visibility_limit=limit,
        interruption_preference="allow",
        focus_mode="automatic",
        layout_stability_weight=0.7,
        entries=entries,
        provenance=provenance(),
        policy_bindings=("attention-policy-v1",),
    )


def item(
    attention_entry,
    *,
    key="item-1",
    role="primary",
    privacy="ordinary",
    redaction="none",
    product="alice",
    host="host-a",
):
    return WorkspaceItemProjection.create(
        item_key=key,
        scope=scope(product, host),
        reference_id=attention_entry.reference_id,
        item_type=(
            "result_capsule"
            if attention_entry.subject_type == "result_capsule"
            else "mission_node"
        ),
        mission_id=attention_entry.mission_id,
        node_id=attention_entry.node_id,
        capsule_id=attention_entry.capsule_id,
        attention_entry_id=attention_entry.entry_id,
        attention_rank=attention_entry.rank,
        role=role,
        privacy_class=privacy,
        redaction_state=redaction,
        state_digest=attention_entry.state_digest,
        projected_metadata_digest=digest(f"projection-{key}"),
        policy_bindings=("workspace-projection-v1",),
    )


def layout(items, *, total=None, product="alice", host="host-a", max_visible=10):
    total_count = len(items) if total is None else total
    omitted = tuple(
        digest(f"omitted-{index}")
        for index in range(total_count - len(items))
    )
    return WorkspaceLayout.create(
        layout_key="layout-1",
        scope=scope(product, host),
        created_at="2026-08-01T10:11:00Z",
        visible_count=len(items),
        total_candidate_count=total_count,
        max_visible=max_visible,
        layout_locked=False,
        stability_anchor_digest=digest("layout-anchor"),
        item_order=tuple(value.item_id for value in items),
        omitted_reference_digests=omitted,
        policy_bindings=("adaptive-compositor-v1",),
    )
