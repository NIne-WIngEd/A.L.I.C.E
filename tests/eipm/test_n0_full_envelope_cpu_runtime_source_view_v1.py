from __future__ import annotations

import json
from pathlib import Path


ROOT=Path(__file__).resolve().parents[2]


def test_cpu_runtime_qualifier_exercises_registered_additional_view_source_text_adapter() -> None:
    config=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_full_envelope_cpu_runtime_qualification_v1.json").read_text()
    )
    qualification=config["qualification"]
    case=config["runtime_case"]
    surfaces=set(config["text_surface_virtualization_stress"]["minimum_virtualized_surfaces"])

    assert qualification["additional_view_source_text_adapter_required"] is True
    assert "additional_view_source" in surfaces
    assert len(case["additional_view_sources"]) == len(case["additional_views"])
    assert all(str(x).strip() for x in case["additional_view_sources"])

    source=(
        ROOT/"scripts/eipm/n0/qualify_n0_v02_full_envelope_cpu_runtime_v1.py"
    ).read_text()
    assert '"additional_view_source_input_ids"' in source
    assert '"additional_view_source_attention_mask"' in source
    assert '"additional_source_views": additional_source_views' not in source
    assert '"additional_view_source": bool(' in source
