from cognitive_kernel import ProductHostScope, ProvenanceReference, canonical_sha256

def scope(product="alice", host="host-a"):
    return ProductHostScope.create(product_id=product, host_instance_id=host, schema_version="1.0.0", encryption_domain=f"private-{host}")

def provenance():
    return ProvenanceReference.create(provenance_type="derived_inference", source_reference_ids=("source-1",), derivation_activity_id="derive-1", responsible_component="synthetic-planner", confidence=0.8)

def digest(label):
    return canonical_sha256({"label": label})
