from scripts.eipm.n0.diagnose_n0_v02_parent_value_path_v0_1 import contains_value, field_indices


def test_contains_value_uses_token_boundaries_for_numeric_values() -> None:
    assert contains_value("worker quota is 4", "4")
    assert not contains_value("worker quota is 34", "4")
    assert contains_value("control epoch is 41.", "41")


def test_contains_value_supports_multiword_values_case_insensitively() -> None:
    assert contains_value("Snapshot interval is 12 minutes.", "12 minutes")
    assert contains_value("SERVICE PROFILE is Burst.", "burst")


def test_field_indices_distinguish_target_and_foil_fields() -> None:
    fields = [
        {"text": "Evidence record A lists secondary for Matrix Osprey's routing lane."},
        {"text": "Evidence record B lists primary for Matrix Osprey's routing lane."},
    ]
    assert field_indices(fields, "primary") == [1]
    assert field_indices(fields, "secondary") == [0]
