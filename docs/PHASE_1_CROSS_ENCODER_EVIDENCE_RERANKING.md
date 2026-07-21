# P1.11 Cross-Encoder Evidence Reranking

The mismatch-only support audit found 19 claims: 8 supported and 11 unsupported.
Three cases had zero supported claims, which points to passage selection as the
primary problem.

This patch keeps the P1.10 source set unchanged, but reranks candidate passages
inside each selected source:

E5/lexical prefilter -> top 12 passages -> MS MARCO Cross-Encoder ->
top 3 non-redundant passages -> grounded generation.

Model:
cross-encoder/ms-marco-MiniLM-L6-v2

The public model is downloaded locally into C:\ALICE_Vault\models\rerankers.
Private text is never uploaded. Response evaluation advances to v8.
