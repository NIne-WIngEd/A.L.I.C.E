from __future__ import annotations

import pytest

from alice_conversation.cli_runtime import (
    ConversationCliCancelledError,
    ConversationCliError,
    ConversationCliInterruptedError,
)
from alice_conversation.state_inspection import inspect_conversation_session

from _cli_helpers import INTERRUPT, make_runtime


def test_interrupted_turn_can_resume(tmp_path):
    runtime, _, store = make_runtime(
        tmp_path,
        [INTERRUPT, "Consider the resumed result."],
    )
    with pytest.raises(ConversationCliInterruptedError):
        runtime.send("Start")
    assert runtime.inspect().turn_statuses == (("interrupted", 1),)
    output = runtime.resume()
    assert output.content == "Consider the resumed result."
    assert runtime.inspect().turn_statuses == (("completed", 1),)
    raw = inspect_conversation_session(
        store,
        session_id=runtime._session_id,
        include_content=True,
    )
    assert len(raw.turns[0].generations) == 2
    assert tuple(g.status for g in raw.turns[0].generations) == (
        "interrupted",
        "completed",
    )


def test_cancel_interrupted_turn(tmp_path):
    runtime, _, _ = make_runtime(tmp_path, [INTERRUPT])
    with pytest.raises(ConversationCliInterruptedError):
        runtime.send("Start")
    assert runtime.cancel() is True
    assert runtime.inspect().turn_statuses == (("cancelled", 1),)
    assert runtime.cancel() is False


def test_resume_requires_interrupted_turn(tmp_path):
    runtime, _, _ = make_runtime(tmp_path, [])
    with pytest.raises(ConversationCliError):
        runtime.resume()


def test_close_rejects_interrupted_turn(tmp_path):
    runtime, _, _ = make_runtime(tmp_path, [INTERRUPT])
    with pytest.raises(ConversationCliInterruptedError):
        runtime.send("Start")
    with pytest.raises(ConversationCliError):
        runtime.close_session()


def test_cancel_allows_close_after_interruption(tmp_path):
    runtime, _, _ = make_runtime(tmp_path, [INTERRUPT])
    with pytest.raises(ConversationCliInterruptedError):
        runtime.send("Start")
    runtime.cancel()
    summary = runtime.close_session()
    assert summary.purged is True


def test_keyboard_interrupt_is_recorded_as_cancelled(tmp_path):
    runtime, model, _ = make_runtime(tmp_path, [KeyboardInterrupt()])
    with pytest.raises(ConversationCliCancelledError) as error:
        runtime.send("Start")
    assert error.value.code == "cli_keyboard_interrupt"
    assert runtime.inspect().turn_statuses == (("cancelled", 1),)
