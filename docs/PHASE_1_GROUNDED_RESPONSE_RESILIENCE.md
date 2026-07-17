# P1.11 — Grounded Response Evaluation Resilience

This patch addresses long local benchmark runs that can fail when one Ollama
request exceeds the HTTP timeout.

Changes:

- increases the per-request timeout to 600 seconds;
- retries timeout failures twice with bounded backoff;
- sends `keep_alive: "30m"` so Ollama keeps `qwen3:8b` loaded between cases;
- caps structured response generation at 512 output tokens;
- loads the local semantic embedding model once per evaluation run;
- saves a private checkpoint after every completed benchmark case;
- resumes completed cases when the same evaluation command is run again;
- prints completed/resumed progress without printing private answers.

Ollama remains loopback-only. Memory writes, tools, web access, and external
actions remain disabled.
