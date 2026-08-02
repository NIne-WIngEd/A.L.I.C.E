from __future__ import annotations

import pytest

from cognitive_kernel import CognitiveKernelContractError

from release_attestation_helpers import make_release


def test_distinct_release_identities_are_isolated() -> None:
    first = make_release(release_id="friday-0.1.0-canary")
    second = make_release(release_id="friday-0.1.1-canary")
    first.assert_isolated_from(second)


def test_same_product_and_release_identity_is_rejected() -> None:
    first = make_release()
    second = make_release()
    with pytest.raises(CognitiveKernelContractError, match="same product"):
        first.assert_isolated_from(second)


def test_alice_and_friday_release_namespaces_are_distinct() -> None:
    friday = make_release(product_id="friday", release_id="candidate-1")
    alice = make_release(product_id="alice", release_id="candidate-1")
    friday.assert_isolated_from(alice)
