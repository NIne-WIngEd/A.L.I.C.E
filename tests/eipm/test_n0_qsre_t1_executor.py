from __future__ import annotations

import unittest

import torch

from alice_personality.n0.qsre_mechanics import QSRE_DEFER, QSRE_FALLBACK, QSRE_RELATIONAL
from alice_personality.n0.qsre_t1_executor import (
    QSRET1Config,
    QSRET1Executor,
    QSRET1OracleOperator,
)


def make_model() -> QSRET1Executor:
    torch.manual_seed(20260920)
    model = QSRET1Executor(
        QSRET1Config(
            field_state_dim=8,
            field_metadata_dim=3,
            edge_metadata_dim=2,
            operator_context_dim=5,
            model_dim=16,
            num_relations=7,
            num_roles=4,
            num_operations=5,
            dropout=0.0,
        )
    )
    model.eval()
    return model


def make_batch(*, fields: int = 5, edges: int = 6, batch: int = 2):
    torch.manual_seed(7)
    field_state = torch.randn(batch, fields, 8)
    field_metadata = torch.randn(batch, fields, 3)
    field_valid = torch.ones(batch, fields, dtype=torch.bool)

    edge_index = torch.zeros(batch, edges, 2, dtype=torch.long)
    for b in range(batch):
        for e in range(edges):
            edge_index[b, e, 0] = e % fields
            edge_index[b, e, 1] = (e + 1) % fields
    edge_relation = torch.tensor(
        [[1 + (e % 3) for e in range(edges)] for _ in range(batch)],
        dtype=torch.long,
    )
    edge_metadata = torch.randn(batch, edges, 2)
    edge_valid = torch.ones(batch, edges, dtype=torch.bool)

    field_support = torch.zeros(batch, fields)
    edge_support = torch.zeros(batch, edges)
    edge_support[:, : min(3, edges)] = 1.0

    relation_sequence = torch.tensor([[1, 2], [1, 2]], dtype=torch.long)
    relation_mask = torch.ones(batch, 2, dtype=torch.bool)
    role_id = torch.tensor([0, 1], dtype=torch.long)
    operation_id = torch.tensor([1, 1], dtype=torch.long)
    focus = torch.zeros(batch, fields)
    focus[:, 0] = 1.0
    context = torch.randn(batch, 5)
    applicability = torch.tensor([0.9, 0.1])

    operator = QSRET1OracleOperator(
        relation_sequence_id=relation_sequence,
        relation_sequence_mask=relation_mask,
        role_id=role_id,
        operation_id=operation_id,
        focus_field_weight=focus,
        context=context,
        applicability=applicability,
    )
    return dict(
        field_state=field_state,
        field_metadata=field_metadata,
        field_valid_mask=field_valid,
        edge_index=edge_index,
        edge_relation_id=edge_relation,
        edge_metadata=edge_metadata,
        edge_valid_mask=edge_valid,
        field_support_weight=field_support,
        edge_support_weight=edge_support,
        operator=operator,
    )


class QsreT1ForwardTests(unittest.TestCase):
    def test_forward_support_local_and_control_separation(self) -> None:
        model = make_model()
        batch = make_batch()
        with torch.no_grad():
            out = model(**batch)

        support = out["support_field_mask"]
        prob = out["relational_probability"]
        self.assertEqual(out["control_state"].tolist(), [QSRE_RELATIONAL, QSRE_FALLBACK])
        self.assertTrue(torch.all(prob[~support] == 0))
        torch.testing.assert_close(prob[0].sum(), torch.tensor(1.0))
        torch.testing.assert_close(prob[1].sum(), torch.tensor(0.0))

    def test_defer_has_no_relational_assertion(self) -> None:
        model = make_model()
        batch = make_batch(batch=1)
        op = batch["operator"]
        batch["operator"] = QSRET1OracleOperator(
            relation_sequence_id=op.relation_sequence_id[:1],
            relation_sequence_mask=op.relation_sequence_mask[:1],
            role_id=op.role_id[:1],
            operation_id=op.operation_id[:1],
            focus_field_weight=op.focus_field_weight[:1],
            context=op.context[:1],
            applicability=torch.tensor([0.5]),
        )
        for key in (
            "field_state","field_metadata","field_valid_mask","edge_index",
            "edge_relation_id","edge_metadata","edge_valid_mask",
            "field_support_weight","edge_support_weight",
        ):
            batch[key] = batch[key][:1]
        with torch.no_grad():
            out = model(**batch)
        self.assertEqual(int(out["control_state"][0]), QSRE_DEFER)
        torch.testing.assert_close(out["relational_probability"].sum(), torch.tensor(0.0))

    def test_edge_permutation_invariance(self) -> None:
        model = make_model()
        batch = make_batch(batch=1)
        for key in (
            "field_state","field_metadata","field_valid_mask","edge_index",
            "edge_relation_id","edge_metadata","edge_valid_mask",
            "field_support_weight","edge_support_weight",
        ):
            batch[key] = batch[key][:1]
        op = batch["operator"]
        batch["operator"] = QSRET1OracleOperator(
            relation_sequence_id=op.relation_sequence_id[:1],
            relation_sequence_mask=op.relation_sequence_mask[:1],
            role_id=op.role_id[:1],
            operation_id=op.operation_id[:1],
            focus_field_weight=op.focus_field_weight[:1],
            context=op.context[:1],
            applicability=torch.tensor([0.9]),
        )
        with torch.no_grad():
            base = model(**batch)["relational_probability"]

        perm = torch.tensor([3, 0, 5, 2, 1, 4])
        permuted = dict(batch)
        for key in ("edge_index","edge_relation_id","edge_metadata","edge_valid_mask","edge_support_weight"):
            permuted[key] = batch[key][:, perm]
        with torch.no_grad():
            changed = model(**permuted)["relational_probability"]
        torch.testing.assert_close(base, changed, atol=1e-6, rtol=1e-6)

    def test_field_permutation_equivariance(self) -> None:
        model = make_model()
        batch = make_batch(batch=1)
        for key in (
            "field_state","field_metadata","field_valid_mask","edge_index",
            "edge_relation_id","edge_metadata","edge_valid_mask",
            "field_support_weight","edge_support_weight",
        ):
            batch[key] = batch[key][:1]
        op = batch["operator"]
        batch["operator"] = QSRET1OracleOperator(
            relation_sequence_id=op.relation_sequence_id[:1],
            relation_sequence_mask=op.relation_sequence_mask[:1],
            role_id=op.role_id[:1],
            operation_id=op.operation_id[:1],
            focus_field_weight=op.focus_field_weight[:1],
            context=op.context[:1],
            applicability=torch.tensor([0.9]),
        )
        with torch.no_grad():
            base = model(**batch)["relational_probability"]

        perm = torch.tensor([2, 4, 0, 3, 1])
        inv = torch.argsort(perm)
        changed = dict(batch)
        for key in ("field_state","field_metadata","field_valid_mask","field_support_weight"):
            changed[key] = batch[key][:, perm]
        changed_edge_index = inv[batch["edge_index"]]
        changed["edge_index"] = changed_edge_index
        changed["operator"] = QSRET1OracleOperator(
            relation_sequence_id=batch["operator"].relation_sequence_id,
            relation_sequence_mask=batch["operator"].relation_sequence_mask,
            role_id=batch["operator"].role_id,
            operation_id=batch["operator"].operation_id,
            focus_field_weight=batch["operator"].focus_field_weight[:, perm],
            context=batch["operator"].context,
            applicability=batch["operator"].applicability,
        )
        with torch.no_grad():
            permuted_prob = model(**changed)["relational_probability"]
        restored = permuted_prob[:, inv]
        torch.testing.assert_close(base, restored, atol=1e-6, rtol=1e-6)

    def test_relation_sequence_order_changes_execution_state(self) -> None:
        model = make_model()
        batch = make_batch(batch=1)
        for key in (
            "field_state","field_metadata","field_valid_mask","edge_index",
            "edge_relation_id","edge_metadata","edge_valid_mask",
            "field_support_weight","edge_support_weight",
        ):
            batch[key] = batch[key][:1]
        op = batch["operator"]
        op_a = QSRET1OracleOperator(
            relation_sequence_id=torch.tensor([[1, 2]]),
            relation_sequence_mask=torch.tensor([[True, True]]),
            role_id=op.role_id[:1],
            operation_id=op.operation_id[:1],
            focus_field_weight=op.focus_field_weight[:1],
            context=op.context[:1],
            applicability=torch.tensor([0.9]),
        )
        op_b = QSRET1OracleOperator(
            relation_sequence_id=torch.tensor([[2, 1]]),
            relation_sequence_mask=torch.tensor([[True, True]]),
            role_id=op.role_id[:1],
            operation_id=op.operation_id[:1],
            focus_field_weight=op.focus_field_weight[:1],
            context=op.context[:1],
            applicability=torch.tensor([0.9]),
        )
        with torch.no_grad():
            a = model(**{**batch, "operator": op_a})["node_state"]
            b = model(**{**batch, "operator": op_b})["node_state"]
        self.assertFalse(torch.allclose(a, b))

    def test_dynamic_field_edge_and_path_shapes(self) -> None:
        model = make_model()
        fields, edges, steps = 70, 300, 4
        batch = make_batch(fields=fields, edges=edges, batch=1)
        for key in (
            "field_state","field_metadata","field_valid_mask","edge_index",
            "edge_relation_id","edge_metadata","edge_valid_mask",
            "field_support_weight","edge_support_weight",
        ):
            batch[key] = batch[key][:1]
        batch["edge_support_weight"][:] = 0
        batch["edge_support_weight"][:, :20] = 1.0
        op0 = batch["operator"]
        batch["operator"] = QSRET1OracleOperator(
            relation_sequence_id=torch.tensor([[1, 2, 3, 1]]),
            relation_sequence_mask=torch.ones(1, steps, dtype=torch.bool),
            role_id=op0.role_id[:1],
            operation_id=op0.operation_id[:1],
            focus_field_weight=op0.focus_field_weight[:1],
            context=op0.context[:1],
            applicability=torch.tensor([0.9]),
        )
        with torch.no_grad():
            out = model(**batch)
        self.assertEqual(out["node_state"].shape[:2], (1, fields))
        self.assertEqual(out["last_edge_state"].shape[:2], (1, edges))

    def test_parameter_scope_is_t1_only(self) -> None:
        model = make_model()
        report = model.parameter_report()
        self.assertGreater(report["total_parameters"], 0)
        self.assertEqual(report["total_parameters"], report["trainable_parameters"])


if __name__ == "__main__":
    unittest.main()
