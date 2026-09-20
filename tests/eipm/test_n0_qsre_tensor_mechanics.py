from __future__ import annotations

import unittest

import torch

from alice_personality.n0.qsre_mechanics import (
    QSRE_DEFER,
    QSRE_FALLBACK,
    QSRE_RELATIONAL,
    QSREQueryOperatorState,
    QSRETensorInputs,
    masked_sparsemax,
    qsre_control_state,
    role_incidence,
    support_cardinality,
    support_local_field_mask,
    support_local_softmax,
)


class QsreTensorMechanicsTests(unittest.TestCase):
    def test_dynamic_tensor_contract_exceeds_old_operating_hints(self) -> None:
        b, l, tq, f, tf, e, ds = 1, 5, 7, 257, 3, 513, 8
        edge_index = torch.zeros(b, e, 2, dtype=torch.long)
        edge_index[0, :, 0] = torch.arange(e) % f
        edge_index[0, :, 1] = (torch.arange(e) * 7 + 1) % f

        inputs = QSRETensorInputs(
            query_hidden_states=torch.zeros(b, l, tq, ds),
            query_token_mask=torch.ones(b, tq, dtype=torch.bool),
            field_token_states=torch.zeros(b, f, tf, ds),
            field_token_mask=torch.ones(b, f, tf, dtype=torch.bool),
            field_base_state=torch.zeros(b, f, 11),
            field_struct_state=torch.zeros(b, f, 13),
            field_metadata=torch.zeros(b, f, 17),
            field_valid_mask=torch.ones(b, f, dtype=torch.bool),
            edge_index=edge_index,
            edge_relation_state=torch.zeros(b, e, 19),
            edge_metadata=torch.zeros(b, e, 23),
            edge_valid_mask=torch.ones(b, e, dtype=torch.bool),
        )
        shape = inputs.validate()
        self.assertEqual(shape["fields"], 257)
        self.assertEqual(shape["edges"], 513)

        operator = QSREQueryOperatorState(
            continuous=torch.zeros(b, 29),
            relation_anchor=torch.zeros(b, 37),
            role_anchor=torch.zeros(b, 6),
            temporal_status=torch.zeros(b, 5),
            applicability=torch.ones(b),
            uncertainty=torch.zeros(b, 4),
            layer_weight=torch.full((b, l), 1.0 / l),
        )
        op_shape = operator.validate(batch=b, layers=l)
        self.assertEqual(op_shape["relation_anchors"], 37)
        self.assertEqual(op_shape["role_anchors"], 6)

    def test_sparsemax_variable_cardinality_and_mask(self) -> None:
        logits = torch.tensor([
            [3.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [100.0, 1.0, 1.0],
        ])
        mask = torch.tensor([
            [True, True, True],
            [True, True, True],
            [True, True, True],
            [False, True, True],
        ])
        out = masked_sparsemax(logits, mask)
        expected = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [1.0 / 3, 1.0 / 3, 1.0 / 3],
            [0.0, 0.5, 0.5],
        ])
        torch.testing.assert_close(out, expected)
        self.assertEqual(support_cardinality(out).tolist(), [1, 2, 3, 2])

    def test_sparsemax_empty_mask_fails_closed(self) -> None:
        logits = torch.tensor([[5.0, 4.0]])
        mask = torch.tensor([[False, False]])
        out = masked_sparsemax(logits, mask)
        torch.testing.assert_close(out, torch.zeros_like(logits))

    def test_source_target_incidence_is_explicit(self) -> None:
        edges = torch.tensor([[[0, 1], [2, 1], [1, 3]]])
        valid = torch.tensor([[True, True, True]])
        source, target = role_incidence(edges, valid, num_fields=4)
        self.assertTrue(source[0, 0, 0])
        self.assertTrue(target[0, 1, 0])
        self.assertTrue(source[0, 2, 1])
        self.assertTrue(target[0, 1, 1])
        self.assertFalse(torch.equal(source, target))

    def test_support_local_readout_excludes_global_distractor(self) -> None:
        logits = torch.tensor([[0.2, 100.0, 0.7, -4.0]])
        field_valid = torch.ones(1, 4, dtype=torch.bool)
        field_support = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        edge_index = torch.zeros(1, 0, 2, dtype=torch.long)
        edge_support = torch.zeros(1, 0)
        edge_valid = torch.zeros(1, 0, dtype=torch.bool)

        mask = support_local_field_mask(
            field_valid, field_support, edge_index, edge_support, edge_valid
        )
        probs = support_local_softmax(logits, mask)
        self.assertEqual(mask.tolist(), [[True, False, True, False]])
        self.assertEqual(float(probs[0, 1]), 0.0)
        self.assertEqual(float(probs[0, 3]), 0.0)
        self.assertEqual(int(probs.argmax(dim=-1)[0]), 2)

    def test_active_edge_adds_only_its_endpoints_to_support(self) -> None:
        field_valid = torch.ones(1, 3, dtype=torch.bool)
        field_support = torch.zeros(1, 3)
        edges = torch.tensor([[[0, 1], [1, 2]]])
        edge_support = torch.tensor([[1.0, 0.0]])
        edge_valid = torch.tensor([[True, True]])
        mask = support_local_field_mask(
            field_valid, field_support, edges, edge_support, edge_valid
        )
        self.assertEqual(mask.tolist(), [[True, True, False]])

    def test_control_state_is_separate_from_support_identity(self) -> None:
        applicability = torch.tensor([0.9, 0.1, 0.5, 0.9])
        has_support = torch.tensor([True, True, True, False])
        state = qsre_control_state(applicability, has_support)
        self.assertEqual(
            state.tolist(),
            [QSRE_RELATIONAL, QSRE_FALLBACK, QSRE_DEFER, QSRE_FALLBACK],
        )

    def test_field_permutation_preserves_semantic_distribution(self) -> None:
        logits = torch.tensor([[0.2, 100.0, 0.7, -4.0]])
        support = torch.tensor([[True, False, True, False]])
        base = support_local_softmax(logits, support)

        perm = torch.tensor([2, 0, 3, 1])
        inv = torch.argsort(perm)
        permuted = support_local_softmax(logits[:, perm], support[:, perm])
        restored = permuted[:, inv]
        torch.testing.assert_close(base, restored)


if __name__ == "__main__":
    unittest.main()
