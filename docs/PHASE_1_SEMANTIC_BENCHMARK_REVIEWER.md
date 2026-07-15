# P1.9 — Private Semantic Benchmark Reviewer

Source hashes are machine identifiers, not review evidence. This reviewer shows
each candidate's local filename, path, ranks, and content preview before asking
the owner to select the correct source.

Run:

```powershell
py scripts\review_semantic_benchmark.py `
  --vault "C:\ALICE_Vault" `
  --benchmark $BenchmarkPath `
  --pilot-name "pilot-v1"
```

Selections:

```text
1       approve candidate 1
1,3     approve candidates 1 and 3
x       exclude the question
s       leave it pending
q       save and quit
```

The script creates a private backup and saves after every answer. Questions,
previews, paths, and source hashes must remain outside Git.
