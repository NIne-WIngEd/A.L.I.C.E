from __future__ import annotations

import pytest

from alice_conversation.cli import run_interactive_cli

from _cli_helpers import make_runtime


def run_lines(runtime, lines, *, grounding_file=None, repository_root=None):
    iterator = iter(lines)
    output = []
    code = run_interactive_cli(
        runtime,
        input_fn=lambda _: next(iterator),
        output_fn=output.append,
        grounding_file=grounding_file,
        repository_root=repository_root,
    )
    return code, output


def test_help_and_exit_commands(tmp_path):
    runtime, _, _ = make_runtime(tmp_path, [])
    code, output = run_lines(runtime, [":help", ":exit"])
    assert code == 0
    assert any(":inspect" in line for line in output)
    assert output[-1] == "A.L.I.C.E. local runtime stopped."


def test_normal_message_is_rendered(tmp_path):
    runtime, _, _ = make_runtime(tmp_path, ["Consider the next step."])
    code, output = run_lines(runtime, ["Hello", ":exit"])
    assert code == 0
    assert "A.L.I.C.E.> Consider the next step." in output


def test_inspect_does_not_print_internal_identifiers(tmp_path):
    runtime, _, _ = make_runtime(tmp_path, ["Consider the next step."])
    _, output = run_lines(runtime, ["Hello", ":inspect", ":exit"])
    rendered = "\n".join(output)
    assert "session-" not in rendered
    assert "turn-" not in rendered
    assert "request-" not in rendered
    assert "Private user text" not in rendered
    assert "Session diagnostics:" in rendered


def test_new_retained_session_command(tmp_path):
    runtime, _, _ = make_runtime(tmp_path, [])
    _, output = run_lines(runtime, [":new retained", ":inspect", ":exit"])
    assert "New retained session started." in output
    assert "  retention: retained" in output


def test_close_then_new_session(tmp_path):
    runtime, _, _ = make_runtime(tmp_path, [])
    _, output = run_lines(runtime, [":close", ":new", ":exit"])
    assert any(line.startswith("Session closed and purged") for line in output)
    assert "New session_only session started." in output


def test_cancel_without_nonterminal_turn(tmp_path):
    runtime, _, _ = make_runtime(tmp_path, [])
    _, output = run_lines(runtime, [":cancel", ":exit"])
    assert "No nonterminal turn was available to cancel." in output


def test_grounding_status_command(tmp_path):
    runtime, _, _ = make_runtime(tmp_path, [])
    _, output = run_lines(runtime, [":grounding", ":exit"])
    assert "Grounding is disabled." in output


def test_unknown_command_is_rejected(tmp_path):
    runtime, _, _ = make_runtime(tmp_path, [])
    _, output = run_lines(runtime, [":tool run", ":exit"])
    assert "Unknown command. Type :help." in output


@pytest.mark.parametrize("command", [":web", ":memory", ":send", ":email", ":shell"])
def test_no_tool_or_action_commands_exist(tmp_path, command):
    runtime, _, _ = make_runtime(tmp_path, [])
    _, output = run_lines(runtime, [command, ":exit"])
    assert "Unknown command. Type :help." in output
