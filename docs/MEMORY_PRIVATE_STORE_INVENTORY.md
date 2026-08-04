# Memory Private-Store Inventory

**Baseline commit:** `d5d311ec49f1e4a3e5a7cf688062c7dc2f46d4ec`<br>
**Original baseline report SHA-256:** `23502F2D024A994C293E1F3476E3873924E79C9A9D325DBFAE0BBAE5AA9FC26D`<br>
**Schema-classification report SHA-256:** `0675CE5D67011CD7522824CAA0A8AA4FCFB827D23276FBF6700A6A1C16FAB9C2`<br>
**Refined:** `2026-08-04T05:23:43Z`<br>
**Privacy mode:** Sanitized schema metadata only

## Safety guarantees

- No private row values or payload content were read.
- No private file path is recorded in this document or either report.
- Store scope identifiers and encryption-domain values remain withheld.
- Public table names are shown only when they match a schema already
  published in this repository.
- Unknown table names are represented only by SHA-256 values in the private
  refinement report.
- SQLite inspection used read-only/query-only access.
- The audit did not modify private stores.

## Vault summary

- Total files: **2861**
- Total bytes: **7.73 GiB**
- Filesystem scan errors: **0**
- SQLite candidates: **6**

## Classification summary

| Classification | Stores |
|---|---:|
| `other_sqlite_store` | 6 |

## Sanitized store records

| Store ID | Classification | Status | Main bytes | WAL bytes | Known public tables | Unknown tables | Schema SHA-256 |
|---|---|---|---:|---:|---:|---:|---|
| `PRIVATE-STORE-001` | `other_sqlite_store` | `schema_metadata_read_only` | 122880 | 0 | 0 | 8 | `78BF71B01B2D9EF90BE41E6305D60C92D1C7694720E655D5A8BBE41957FE9AD3` |
| `PRIVATE-STORE-002` | `other_sqlite_store` | `schema_metadata_read_only` | 143360 | 0 | 0 | 8 | `7157E1255C6E5AA4B164941EC99CA7F6A9FDC7E1F1AB37C99703B926D8067D2F` |
| `PRIVATE-STORE-003` | `other_sqlite_store` | `schema_metadata_read_only` | 143360 | 0 | 0 | 8 | `5E16A66EFCCA0FFAF0AE5AE693C5FE3D4FB602E6DAD33F68CD3C9CE3C1B4923C` |
| `PRIVATE-STORE-004` | `other_sqlite_store` | `schema_metadata_read_only` | 2256896 | 0 | 0 | 3 | `B4F495FA93C2D022D4C46F626F6A99FB6E923B980FA2E0EC0824436ABFE6B029` |
| `PRIVATE-STORE-005` | `other_sqlite_store` | `schema_metadata_read_only` | 28119040 | 0 | 0 | 9 | `92E6CB211AF56B1628944D2258D4FAE5A9EA16129C7F8D41B68DC46E93B97A5A` |
| `PRIVATE-STORE-006` | `other_sqlite_store` | `schema_metadata_read_only` | 56406016 | 0 | 0 | 7 | `BE6E8403E36A287B7229F6E206E08AAD1BF2F25085B7814DA1140AA2A6D2D514` |

## Interpretation

A file is called an A.L.I.C.E. memory/storage store only when its schema
matches a table signature already published in the repository. Other
SQLite files remain unclassified and are not assumed to be memory
authorities.

This refinement corrects the original prefix-based classifier, which did
not recognize the published Phase 2 root table name `memories` and could
therefore under-classify valid Phase 2 stores.

## Limitations

- Schema matching proves store type, not correctness or active use.
- Row counts do not reveal whether records are current, historical, test,
  or migration artifacts.
- Encrypted or locked databases may remain metadata unavailable.
- M1 still needs a canonical private-store registry with product, host,
  encryption-domain, authority, derivative, and deletion roles.
