# P1.11 Structured Output Budget Fix

The first v5 benchmark attempt stopped on the first case because Ollama returned
JSON that ended in the middle of a string.

The grounded-response policy limited `num_predict` to 512 tokens. The response
schema duplicates answer content into structured claim objects and may also
include uncertainty and contradiction notes, so 512 tokens can be too small for
the complete JSON document.

This patch raises `maximum_output_tokens` from 512 to 2048.

The change increases only the maximum generation budget. It does not require the
model to use all 2048 tokens, does not weaken the JSON schema or citation
verifier, and does not enable memory writes, tools, web access, or actions.
