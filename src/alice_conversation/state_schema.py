"""SQLite schema contract for private A.L.I.C.E. conversation state."""

from __future__ import annotations

SCHEMA_VERSION = 1

SESSION_STATUSES = ("active", "interrupted", "completed")
SESSION_RETENTIONS = ("session_only", "retained")
TURN_STATUSES = (
    "received",
    "context_ready",
    "generating",
    "interrupted",
    "completed",
    "cancelled",
    "failed",
)
GENERATION_STATUSES = (
    "started",
    "interrupted",
    "completed",
    "cancelled",
    "failed",
)
REFERENCE_KINDS = (
    "memory",
    "memory_source",
    "phase1_chunk",
    "phase1_source",
    "grounding_packet",
)
REASONING_STATUSES = (
    "not_requested",
    "not_persisted",
    "provider_hidden",
    "unavailable",
)
VALIDATION_OUTCOMES = (
    "not_evaluated",
    "accepted",
    "rejected",
    "abstained",
)
ORDINARY_CLASSIFICATIONS = ("PUBLIC", "INTERNAL", "PRIVATE")

REQUIRED_TABLES = (
    "conversation_schema_migrations",
    "conversation_sessions",
    "conversation_turns",
    "conversation_messages",
    "conversation_turn_references",
    "conversation_generations",
    "conversation_state_events",
    "conversation_session_tombstones",
)


def _quoted(values: tuple[str, ...]) -> str:
    return ", ".join(f"'{value}'" for value in values)


MIGRATION_1_SQL = f"""
CREATE TABLE conversation_sessions (
    session_id TEXT PRIMARY KEY,
    status TEXT NOT NULL CHECK (status IN ({_quoted(SESSION_STATUSES)})),
    retention TEXT NOT NULL CHECK (retention IN ({_quoted(SESSION_RETENTIONS)})),
    data_classification TEXT NOT NULL
        CHECK (data_classification IN ({_quoted(ORDINARY_CLASSIFICATIONS)})),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    closed_at TEXT
);

CREATE TABLE conversation_turns (
    turn_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL CHECK (turn_index >= 0),
    status TEXT NOT NULL CHECK (status IN ({_quoted(TURN_STATUSES)})),
    grounding_packet_id TEXT,
    grounding_packet_sha256 TEXT,
    interruption_count INTEGER NOT NULL DEFAULT 0 CHECK (interruption_count >= 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    failure_code TEXT,
    FOREIGN KEY (session_id)
        REFERENCES conversation_sessions(session_id)
        ON DELETE CASCADE,
    UNIQUE (session_id, turn_index),
    CHECK (
        (grounding_packet_id IS NULL AND grounding_packet_sha256 IS NULL)
        OR
        (grounding_packet_id IS NOT NULL AND grounding_packet_sha256 IS NOT NULL)
    )
);

CREATE TABLE conversation_messages (
    message_id TEXT PRIMARY KEY,
    turn_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal IN (0, 1)),
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    data_classification TEXT NOT NULL
        CHECK (data_classification IN ({_quoted(ORDINARY_CLASSIFICATIONS)})),
    created_at TEXT NOT NULL,
    FOREIGN KEY (turn_id)
        REFERENCES conversation_turns(turn_id)
        ON DELETE CASCADE,
    UNIQUE (turn_id, ordinal),
    UNIQUE (turn_id, role),
    CHECK (
        (role = 'user' AND ordinal = 0)
        OR
        (role = 'assistant' AND ordinal = 1)
    )
);

CREATE TABLE conversation_turn_references (
    reference_id TEXT PRIMARY KEY,
    turn_id TEXT NOT NULL,
    reference_index INTEGER NOT NULL CHECK (reference_index >= 0),
    source_kind TEXT NOT NULL CHECK (source_kind IN ({_quoted(REFERENCE_KINDS)})),
    source_ref TEXT NOT NULL,
    citation_token TEXT,
    content_sha256 TEXT,
    data_classification TEXT NOT NULL
        CHECK (data_classification IN ({_quoted(ORDINARY_CLASSIFICATIONS)})),
    created_at TEXT NOT NULL,
    FOREIGN KEY (turn_id)
        REFERENCES conversation_turns(turn_id)
        ON DELETE CASCADE,
    UNIQUE (turn_id, reference_index),
    UNIQUE (turn_id, source_kind, source_ref)
);

CREATE TABLE conversation_generations (
    generation_id TEXT PRIMARY KEY,
    turn_id TEXT NOT NULL,
    attempt_index INTEGER NOT NULL CHECK (attempt_index >= 0),
    request_id TEXT NOT NULL UNIQUE,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ({_quoted(GENERATION_STATUSES)})),
    reasoning_status TEXT NOT NULL
        CHECK (reasoning_status IN ({_quoted(REASONING_STATUSES)})),
    validation_outcome TEXT NOT NULL
        CHECK (validation_outcome IN ({_quoted(VALIDATION_OUTCOMES)})),
    finish_reason TEXT,
    response_sha256 TEXT,
    failure_code TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    FOREIGN KEY (turn_id)
        REFERENCES conversation_turns(turn_id)
        ON DELETE CASCADE,
    UNIQUE (turn_id, attempt_index),
    CHECK (status != 'completed' OR response_sha256 IS NOT NULL),
    CHECK (status != 'completed' OR completed_at IS NOT NULL),
    CHECK (status != 'started' OR completed_at IS NULL)
);

CREATE TABLE conversation_state_events (
    event_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_id TEXT,
    event_type TEXT NOT NULL,
    detail_code TEXT,
    occurred_at TEXT NOT NULL,
    FOREIGN KEY (session_id)
        REFERENCES conversation_sessions(session_id)
        ON DELETE CASCADE,
    FOREIGN KEY (turn_id)
        REFERENCES conversation_turns(turn_id)
        ON DELETE CASCADE
);

CREATE TABLE conversation_session_tombstones (
    session_id TEXT PRIMARY KEY,
    retention TEXT NOT NULL CHECK (retention IN ({_quoted(SESSION_RETENTIONS)})),
    deleted_at TEXT NOT NULL,
    turn_count INTEGER NOT NULL CHECK (turn_count >= 0),
    message_count INTEGER NOT NULL CHECK (message_count >= 0),
    reference_count INTEGER NOT NULL CHECK (reference_count >= 0),
    generation_count INTEGER NOT NULL CHECK (generation_count >= 0)
);

CREATE INDEX idx_conversation_turns_session
    ON conversation_turns(session_id, turn_index);
CREATE INDEX idx_conversation_messages_turn
    ON conversation_messages(turn_id, ordinal);
CREATE INDEX idx_conversation_references_turn
    ON conversation_turn_references(turn_id, reference_index);
CREATE INDEX idx_conversation_generations_turn
    ON conversation_generations(turn_id, attempt_index);
CREATE INDEX idx_conversation_events_session
    ON conversation_state_events(session_id, occurred_at);
"""
