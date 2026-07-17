# Phase 1 — Grounded Response Generation

**Subphase:** P1.11  
**Input:** verified P1.10 context packages  
**LLM:** local Ollama `qwen3:8b` by default  
**Memory writes:** forbidden  
**External actions:** forbidden  
**Tool calls:** forbidden  
**Web access:** forbidden

## Purpose

P1.11 is the first layer in which A.L.I.C.E. produces a natural-language answer
about the owner.

The model is not allowed to search memory directly. It receives only the P1.10
evidence package.

## Grounding contract

Every personal factual claim must:

- be represented in the structured `claims` array;
- use `claim_type: fact`;
- cite one or more package-local sources such as `[S1]`.

Every inference must:

- use `claim_type: inference`;
- remain clearly distinguishable from retrieved fact;
- cite the evidence supporting the inference.

The verifier rejects:

- invented citations;
- uncited claims;
- citations outside the P1.10 package;
- silently resolved contradiction groups;
- missing package fingerprints;
- any response package that enables memory writes, tools, web access, or
  external actions.

## Prompt-injection boundary

Retrieved source text is untrusted data. The system prompt explicitly tells the
model never to execute instructions found inside evidence.

P1.11 never exposes tools to the generation model.

## Local Ollama

Default:

```text
model: qwen3:8b
endpoint: http://127.0.0.1:11434/api/generate
think: false
temperature: 0.1
```

The request uses Ollama structured output with a JSON schema.

## Ask A.L.I.C.E.

```powershell
py scripts\answer_grounded.py `
  --vault "C:\ALICE_Vault" `
  --query "What research experience do I have with AFM images?" `
  --show-answer
```

This performs:

```text
Question
  ↓
P1.10 hybrid retrieval + evidence package
  ↓
P1.11 local structured generation
  ↓
Citation verifier
  ↓
Verified answer
```

Without `--show-answer`, only an aggregate non-content summary is printed.

## Private outputs

```text
C:\ALICE_Vault\manifests\responses\pilot-v1\
└── grounded-response-<UUID>.json
```

Aggregate summaries are written under:

```text
C:\ALICE_Vault\manifests\exports\
```

## Evaluation

P1.11 reuses the private P1.9 human-curated benchmark and measures:

- verified response rate;
- expected-source citation rate;
- claim citation coverage.

It does not use another LLM as a judge in this phase.

## Deliberately deferred

- autonomous memory writes;
- memory consolidation;
- general web browsing;
- tool execution;
- email/calendar/filesystem actions;
- long multi-turn conversation state;
- self-modification.
