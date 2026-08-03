from pathlib import Path
import pytest

from cognitive_kernel import UnsafeRawBufferPathError, validate_raw_buffer_root


def test_repository_internal_raw_buffer_path_is_rejected(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    with pytest.raises(UnsafeRawBufferPathError):
        validate_raw_buffer_root(
            repository / "private" / "raw-buffer",
            repository_root=repository,
        )


def test_external_raw_buffer_path_is_accepted(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    external = tmp_path / "vault" / "raw-buffer"
    assert validate_raw_buffer_root(
        external,
        repository_root=repository,
    ) == external.resolve(strict=False)
