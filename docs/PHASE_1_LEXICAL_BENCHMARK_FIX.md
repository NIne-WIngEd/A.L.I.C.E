# P1.8 Lexical Benchmark Schema Fix

The benchmark vocabulary helper is created in SQLite's `temp` schema while the
indexed `chunk_fts` table lives in `main`. The FTS5 vocabulary module therefore
uses the documented three-argument form:

```sql
CREATE VIRTUAL TABLE temp.term_instances
USING fts5vocab(main, 'chunk_fts', 'instance');
```

This prevents SQLite from incorrectly looking for `temp.chunk_fts`. A
file-backed regression test now builds the index and creates a lexical benchmark.
