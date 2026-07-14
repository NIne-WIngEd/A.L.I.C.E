from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from alice_vault.retrieval import (
    SearchFilters,
    build_index,
    create_lexical_benchmark,
    evaluate,
    search_index,
    verify_index,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RetrievalTests(unittest.TestCase):
    def make_vault(self, base: Path) -> Path:
        vault = base / 'vault'
        chunk_root = vault / 'derived' / 'pilot-v1' / 'chunks' / 'chunk-test'
        text_root = chunk_root / 'text'
        text_root.mkdir(parents=True)
        (vault / 'manifests' / 'exports').mkdir(parents=True)
        (vault / 'temporary').mkdir()

        fixtures = [
            ('c1', 'source-a', 0, 'AFM segmentation research used a U-Net model and masks.',
             'research_project', '2026', 'lab', False),
            ('c2', 'source-a', 2, 'The AFM pipeline included classification and Voronoi analysis.',
             'research_project', '2026', 'lab', False),
            ('c3', 'source-b', 0, 'A Vanderbilt transfer plan changed after financial problems.',
             'education', '2026', 'personal', True),
        ]
        records = []
        for chunk_id, source, index, text, family, year, bucket, truncated in fixtures:
            path = text_root / f'{chunk_id}.txt'
            path.write_text(text, encoding='utf-8')
            records.append({
                'chunk_id': chunk_id,
                'source_content_sha256': source,
                'source_text_sha256': 't' * 64,
                'normalized_source_text_sha256': 'n' * 64,
                'chunk_index': index,
                'start_char': 0,
                'end_char': len(text),
                'char_count': len(text),
                'chunk_text_sha256': digest(path),
                'family': family,
                'parser_id': 'test-parser',
                'extraction_registry_digest': 'r' * 64,
                'source_extraction_truncated': truncated,
                'source_extraction_warnings': [],
                'provenance_path_count': 1,
                'provenance': [{
                    'file_id': f'file-{chunk_id}',
                    'original_relative_path': f'{bucket}/{chunk_id}.txt',
                    'filename': f'{chunk_id}.txt',
                    'role': 'primary',
                    'family': family,
                    'source_bucket': bucket,
                    'year_hint': year,
                    'duplicate_control_group': '',
                    'known_contradiction_group': (
                        'education-status' if truncated else ''
                    ),
                }],
            })
        records_path = chunk_root / 'chunk-records.jsonl'
        with records_path.open('w', encoding='utf-8') as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True,
                                        separators=(',', ':')) + '\n')
        manifest = {
            'chunk_set_id': 'chunk-test',
            'manifest_fingerprint': 'm' * 64,
            'chunk_records_sha256': digest(records_path),
            'chunk_count': len(records),
        }
        (chunk_root / 'chunk-manifest.json').write_text(
            json.dumps(manifest), encoding='utf-8'
        )
        return vault

    def test_build_verify_search_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            vault = self.make_vault(Path(temp))
            policy = ROOT / 'policies' / 'retrieval_policy.json'
            summary = build_index(vault_root=vault, policy_path=policy)
            self.assertEqual(summary['chunk_count'], 3)
            self.assertEqual(summary['source_count'], 2)
            verification = verify_index(vault_root=vault, policy_path=policy)
            self.assertTrue(verification['ready_for_evaluation'])
            self.assertEqual(verification['verified_fts_rows'], 3)

            result = search_index(
                vault_root=vault,
                policy_path=policy,
                query='AFM U-Net segmentation',
                filters=SearchFilters(families=('research_project',)),
                limit=5,
            )
            self.assertGreaterEqual(result['result_count'], 1)
            self.assertEqual(result['results'][0]['source_content_sha256'],
                             'source-a')

            filtered = search_index(
                vault_root=vault,
                policy_path=policy,
                query='Vanderbilt financial',
                filters=SearchFilters(
                    years=('2026',),
                    source_buckets=('personal',),
                    include_truncated=False,
                ),
                limit=5,
            )
            self.assertEqual(filtered['result_count'], 0)
            resumed = build_index(vault_root=vault, policy_path=policy)
            self.assertTrue(resumed['resumed_existing_index'])

    def test_tamper_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            vault = self.make_vault(Path(temp))
            policy = ROOT / 'policies' / 'retrieval_policy.json'
            summary = build_index(vault_root=vault, policy_path=policy)
            database = Path(summary['database_path'])
            connection = sqlite3.connect(database)
            connection.execute("UPDATE chunks SET body='tampered' WHERE rowid=1")
            connection.commit()
            connection.close()
            verification = verify_index(vault_root=vault, policy_path=policy)
            self.assertFalse(verification['ready_for_evaluation'])
            self.assertGreater(verification['error_count'], 0)


    def test_lexical_benchmark_reads_main_fts_table(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            vault = self.make_vault(Path(temp))
            policy = ROOT / 'policies' / 'retrieval_policy.json'
            build_index(vault_root=vault, policy_path=policy)
            summary = create_lexical_benchmark(
                vault_root=vault,
                policy_path=policy,
                case_count=5,
            )
            self.assertGreater(summary['created_cases'], 0)
            self.assertTrue(Path(summary['benchmark_path']).is_file())

    def test_evaluation_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            vault = Path(temp)
            benchmark = {
                'benchmark_schema_version': 1,
                'benchmark_id': 'bench-1',
                'benchmark_type': 'lexical_smoke',
                'cases': [
                    {'query_id': 'q1', 'question': 'afm',
                     'expected_source_sha256': ['source-a'],
                     'expected_chunk_ids': [], 'filters': {}},
                    {'query_id': 'q2', 'question': 'vanderbilt',
                     'expected_source_sha256': ['source-b'],
                     'expected_chunk_ids': [], 'filters': {}},
                ],
            }
            benchmark_path = vault / 'benchmark.json'
            benchmark_path.write_text(json.dumps(benchmark), encoding='utf-8')
            responses = [
                {'query_plan': 'and',
                 'results': [{'source_content_sha256': 'source-a'}]},
                {'query_plan': 'and',
                 'results': [
                     {'source_content_sha256': 'other'},
                     {'source_content_sha256': 'source-b'},
                 ]},
            ]
            with patch('alice_vault.retrieval.search_index',
                       side_effect=responses):
                result = evaluate(vault_root=vault,
                                  benchmark_path=benchmark_path)
            self.assertEqual(result['hit_rate_at_k']['1'], 0.5)
            self.assertEqual(result['hit_rate_at_k']['3'], 1.0)
            self.assertEqual(result['mean_reciprocal_rank_at_10'], 0.75)


if __name__ == '__main__':
    unittest.main()
