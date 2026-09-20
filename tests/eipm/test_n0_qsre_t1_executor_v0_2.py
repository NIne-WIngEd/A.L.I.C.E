from __future__ import annotations

import unittest

import torch

from alice_personality.n0.qsre_mechanics import QSRE_FALLBACK
from alice_personality.n0.qsre_t1_executor import (
    QSRET1Config,
    QSRET1Executor,
    QSRET1OracleOperator,
    QSRE_T1_OPERATION_PATH_FOLLOW,
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


def base_batch(fields: int = 5, edges: int = 4) -> dict:
    torch.manual_seed(7)
    return {
        "field_state": torch.randn(1, fields, 8),
        "field_metadata": torch.randn(1, fields, 3),
        "field_valid_mask": torch.ones(1, fields, dtype=torch.bool),
        "edge_index": torch.zeros(1, edges, 2, dtype=torch.long),
        "edge_relation_id": torch.zeros(1, edges, dtype=torch.long),
        "edge_metadata": torch.randn(1, edges, 2),
        "edge_valid_mask": torch.ones(1, edges, dtype=torch.bool),
        "field_support_weight": torch.zeros(1, fields),
        "edge_support_weight": torch.ones(1, edges),
    }


class QsreT1CausalMechanicsV02Tests(unittest.TestCase):
    def test_direct_field_support_is_rejected(self) -> None:
        model = make_model()
        batch = base_batch()
        batch["field_support_weight"][0, 0] = 1.0
        operator = QSRET1OracleOperator(
            relation_sequence_id=torch.tensor([[0]]),
            relation_sequence_mask=torch.tensor([[True]]),
            role_id=torch.tensor([1]),
            operation_id=torch.tensor([0]),
            focus_field_weight=torch.zeros(1, 5),
            context=torch.zeros(1, 5),
            applicability=torch.tensor([0.9]),
        )
        with self.assertRaises(ValueError):
            model(**batch, operator=operator)

    def test_fractional_oracle_edge_support_is_rejected(self) -> None:
        model = make_model()
        batch = base_batch()
        batch["edge_support_weight"][0, 0] = 0.25
        operator = QSRET1OracleOperator(
            relation_sequence_id=torch.tensor([[0]]),
            relation_sequence_mask=torch.tensor([[True]]),
            role_id=torch.tensor([1]),
            operation_id=torch.tensor([0]),
            focus_field_weight=torch.zeros(1, 5),
            context=torch.zeros(1, 5),
            applicability=torch.tensor([0.9]),
        )
        with self.assertRaises(ValueError):
            model(**batch, operator=operator)

    def test_ordered_path_has_exact_structural_frontier(self) -> None:
        model = make_model()
        batch = base_batch(fields=5, edges=4)
        batch["edge_index"][0] = torch.tensor(
            [[0, 1], [1, 2], [0, 3], [3, 4]]
        )
        batch["edge_relation_id"][0] = torch.tensor(
            [1, 2, 2, 1]
        )

        op_a = QSRET1OracleOperator(
            relation_sequence_id=torch.tensor([[1, 2]]),
            relation_sequence_mask=torch.tensor([[True, True]]),
            role_id=torch.tensor([1]),
            operation_id=torch.tensor(
                [QSRE_T1_OPERATION_PATH_FOLLOW]
            ),
            focus_field_weight=torch.tensor(
                [[1.0, 0.0, 0.0, 0.0, 0.0]]
            ),
            context=torch.zeros(1, 5),
            applicability=torch.tensor([0.9]),
        )
        op_b = QSRET1OracleOperator(
            relation_sequence_id=torch.tensor([[2, 1]]),
            relation_sequence_mask=torch.tensor([[True, True]]),
            role_id=torch.tensor([1]),
            operation_id=torch.tensor(
                [QSRE_T1_OPERATION_PATH_FOLLOW]
            ),
            focus_field_weight=torch.tensor(
                [[1.0, 0.0, 0.0, 0.0, 0.0]]
            ),
            context=torch.zeros(1, 5),
            applicability=torch.tensor([0.9]),
        )

        with torch.no_grad():
            a = model(**batch, operator=op_a)
            b = model(**batch, operator=op_b)

        self.assertEqual(
            a["path_frontier"].nonzero(as_tuple=False).tolist(),
            [[0, 2]],
        )
        self.assertEqual(
            b["path_frontier"].nonzero(as_tuple=False).tolist(),
            [[0, 4]],
        )
        self.assertEqual(
            a["readout_field_mask"].nonzero(as_tuple=False).tolist(),
            [[0, 2]],
        )
        self.assertEqual(
            b["readout_field_mask"].nonzero(as_tuple=False).tolist(),
            [[0, 4]],
        )
        torch.testing.assert_close(
            a["relational_probability"][0, 2],
            torch.tensor(1.0),
        )
        torch.testing.assert_close(
            b["relational_probability"][0, 4],
            torch.tensor(1.0),
        )

    def test_nodes_not_incident_to_active_relation_do_not_update(self) -> None:
        model = make_model()
        batch = base_batch(fields=4, edges=2)
        batch["edge_index"][0] = torch.tensor(
            [[0, 1], [2, 3]]
        )
        batch["edge_relation_id"][0] = torch.tensor([1, 2])
        operator = QSRET1OracleOperator(
            relation_sequence_id=torch.tensor([[1]]),
            relation_sequence_mask=torch.tensor([[True]]),
            role_id=torch.tensor([1]),
            operation_id=torch.tensor([0]),
            focus_field_weight=torch.zeros(1, 4),
            context=torch.zeros(1, 5),
            applicability=torch.tensor([0.9]),
        )

        with torch.no_grad():
            initial = (
                model.node_projection(batch["field_state"])
                + model.field_metadata_projection(
                    batch["field_metadata"]
                )
            )
            out = model(**batch, operator=operator)

        torch.testing.assert_close(
            out["node_state"][0, 2:],
            initial[0, 2:],
        )

    def test_dead_path_fails_closed_to_fallback(self) -> None:
        model = make_model()
        batch = base_batch(fields=3, edges=1)
        batch["edge_index"][0] = torch.tensor([[0, 1]])
        batch["edge_relation_id"][0] = torch.tensor([1])
        operator = QSRET1OracleOperator(
            relation_sequence_id=torch.tensor([[2]]),
            relation_sequence_mask=torch.tensor([[True]]),
            role_id=torch.tensor([1]),
            operation_id=torch.tensor(
                [QSRE_T1_OPERATION_PATH_FOLLOW]
            ),
            focus_field_weight=torch.tensor(
                [[1.0, 0.0, 0.0]]
            ),
            context=torch.zeros(1, 5),
            applicability=torch.tensor([0.9]),
        )

        with torch.no_grad():
            out = model(**batch, operator=operator)

        self.assertEqual(
            int(out["control_state"][0]),
            QSRE_FALLBACK,
        )
        torch.testing.assert_close(
            out["relational_probability"].sum(),
            torch.tensor(0.0),
        )

    def test_path_edge_permutation_invariance(self) -> None:
        model = make_model()
        batch = base_batch(fields=5, edges=4)
        batch["edge_index"][0] = torch.tensor(
            [[0, 1], [1, 2], [0, 3], [3, 4]]
        )
        batch["edge_relation_id"][0] = torch.tensor(
            [1, 2, 2, 1]
        )
        operator = QSRET1OracleOperator(
            relation_sequence_id=torch.tensor([[1, 2]]),
            relation_sequence_mask=torch.tensor([[True, True]]),
            role_id=torch.tensor([1]),
            operation_id=torch.tensor(
                [QSRE_T1_OPERATION_PATH_FOLLOW]
            ),
            focus_field_weight=torch.tensor(
                [[1.0, 0.0, 0.0, 0.0, 0.0]]
            ),
            context=torch.zeros(1, 5),
            applicability=torch.tensor([0.9]),
        )

        with torch.no_grad():
            base = model(
                **batch,
                operator=operator,
            )["relational_probability"]

        perm = torch.tensor([3, 0, 2, 1])
        changed = dict(batch)
        for key in (
            "edge_index",
            "edge_relation_id",
            "edge_metadata",
            "edge_valid_mask",
            "edge_support_weight",
        ):
            changed[key] = batch[key][:, perm]

        with torch.no_grad():
            permuted = model(
                **changed,
                operator=operator,
            )["relational_probability"]

        torch.testing.assert_close(
            base,
            permuted,
            atol=1e-6,
            rtol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
