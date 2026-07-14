from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.parser_registry import ParserSpec
from alice_vault.safe_parsers import parse_document


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path)
    args = parser.parse_args()

    try:
        request = json.loads(
            args.request.read_text(encoding="utf-8")
        )
        source = Path(request["source_path"]).resolve(strict=True)
        output_text = Path(request["output_text_path"]).resolve()
        spec = ParserSpec.from_dict(request["parser_spec"])

        expected_size = int(request["expected_size"])
        expected_hash = str(request["expected_sha256"])
        before = source.stat()
        if before.st_size != expected_size:
            raise RuntimeError("Source size does not match pilot manifest")
        if sha256_file(source) != expected_hash:
            raise RuntimeError("Source hash does not match pilot manifest")

        parsed = parse_document(source, spec)

        after = source.stat()
        if (
            after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
        ):
            raise RuntimeError("Source changed during extraction")

        output_text.parent.mkdir(parents=True, exist_ok=True)
        temp_text = output_text.with_name(
            f".{output_text.name}.{os.getpid()}.tmp"
        )
        temp_text.write_text(parsed.text, encoding="utf-8")
        os.replace(temp_text, output_text)

        response = {
            "status": "success",
            "parser_id": spec.parser_id,
            "family": spec.family,
            "source_sha256": expected_hash,
            "source_size_bytes": expected_size,
            "text_chars": len(parsed.text),
            "text_bytes": output_text.stat().st_size,
            "text_sha256": sha256_file(output_text),
            "truncated": parsed.truncated,
            "warnings": parsed.warnings,
            "parser_metadata": parsed.metadata,
        }
        atomic_json(args.response, response)
        return 0

    except Exception as exc:
        atomic_json(
            args.response,
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
