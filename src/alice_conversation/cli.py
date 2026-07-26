"""Interactive local terminal entry point for A.L.I.C.E. Phase 3 P3.7."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Sequence

from .cli_runtime import (
    ConversationCliCancelledError,
    ConversationCliError,
    ConversationCliInterruptedError,
    ConversationCliRuntime,
    ConversationCliTurnError,
    ConversationCliValidationError,
    build_local_conversation_runtime,
)
from .contracts import ConversationContractError
from .grounding_io import load_conversation_grounding_packet

_HELP = """Commands:
  :help                         Show this command list.
  :new [session_only|retained] Start a new private session.
  :close                        Close the current session.
  :inspect                      Show metadata-safe session diagnostics.
  :cancel                       Cancel the current nonterminal turn.
  :resume                       Resume one interrupted turn.
  :grounding                    Show grounding status.
  :grounding off                Disable grounding for future turns.
  :grounding reload             Reload the startup grounding file.
  :exit                         Close the session and exit.
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m alice_conversation.cli",
        description="Run A.L.I.C.E.'s local controlled conversational runtime.",
    )
    parser.add_argument(
        "--repository-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="A.L.I.C.E. repository root.",
    )
    parser.add_argument(
        "--vault-root",
        required=True,
        help="Private vault root outside the repository.",
    )
    parser.add_argument(
        "--provider",
        required=True,
        choices=("ollama-local",),
        help="Explicit local model provider.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Explicit policy-approved local model.",
    )
    parser.add_argument(
        "--retention",
        choices=("session_only", "retained"),
        default="session_only",
        help="Conversation retention mode.",
    )
    parser.add_argument(
        "--grounding-file",
        help="Optional prebuilt grounding-packet JSON outside the repository.",
    )
    return parser


def _render_turn(runtime: ConversationCliRuntime, output_fn: Callable[[str], None], result) -> None:
    prefix = "A.L.I.C.E."
    if result.validation_outcome == "abstained":
        prefix += " [abstained]"
    output_fn(f"{prefix}> {result.content}")
    if result.citation_tokens:
        output_fn("Citations: " + " ".join(result.citation_tokens))


def _render_inspection(runtime: ConversationCliRuntime, output_fn: Callable[[str], None]) -> None:
    value = runtime.inspect()
    statuses = ", ".join(f"{name}={count}" for name, count in value.turn_statuses)
    output_fn("Session diagnostics:")
    output_fn(f"  status: {value.status}")
    output_fn(f"  retention: {value.retention}")
    output_fn(f"  classification: {value.data_classification}")
    output_fn(f"  model: {value.provider}/{value.model}")
    output_fn(f"  turns: {value.turn_count}" + (f" ({statuses})" if statuses else ""))
    output_fn(f"  messages: {value.message_count}")
    output_fn(f"  references: {value.reference_count}")
    output_fn(f"  generations: {value.generation_count}")
    output_fn(f"  last validation: {value.last_validation_outcome or 'none'}")
    output_fn(f"  last failure: {value.last_failure_code or 'none'}")
    output_fn(
        "  grounding: "
        + (
            f"enabled ({value.grounding_outcome}; "
            f"claims={value.grounding_claim_count}; "
            f"citations={value.grounding_citation_count})"
            if value.grounding_enabled
            else "disabled"
        )
    )


def _render_grounding(runtime: ConversationCliRuntime, output_fn: Callable[[str], None]) -> None:
    value = runtime.grounding_status()
    if not value.enabled:
        output_fn("Grounding is disabled.")
        return
    output_fn(
        f"Grounding is enabled: outcome={value.outcome}; "
        f"claims={value.claim_count}; citations={value.citation_count}."
    )


def run_interactive_cli(
    runtime: ConversationCliRuntime,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    repository_root: str | Path | None = None,
    grounding_file: str | Path | None = None,
) -> int:
    output_fn("A.L.I.C.E. local conversation runtime is ready.")
    output_fn("Type :help for commands. No tools, web access, or external actions are enabled.")
    while True:
        try:
            line = input_fn("You> ")
        except EOFError:
            line = ":exit"
        except KeyboardInterrupt:
            output_fn("")
            if runtime.cancel():
                output_fn("The nonterminal turn was cancelled.")
            else:
                output_fn("No nonterminal turn was available to cancel.")
            continue
        text = line.strip()
        if not text:
            continue
        if not text.startswith(":"):
            try:
                _render_turn(runtime, output_fn, runtime.send(text))
            except ConversationCliValidationError as exc:
                codes = ", ".join(exc.issue_codes) or "unspecified"
                output_fn(f"Response rejected [{exc.code}]: {codes}.")
            except ConversationCliInterruptedError as exc:
                output_fn(f"Generation interrupted [{exc.code}]. Use :resume or :cancel.")
            except ConversationCliCancelledError as exc:
                output_fn(f"Generation cancelled [{exc.code}].")
            except ConversationCliTurnError as exc:
                output_fn(f"Generation failed [{exc.code}].")
            except ConversationCliError as exc:
                output_fn(f"Runtime error: {exc}")
            continue

        parts = text.split()
        command = parts[0].lower()
        try:
            if command == ":help" and len(parts) == 1:
                output_fn(_HELP.rstrip())
            elif command == ":inspect" and len(parts) == 1:
                _render_inspection(runtime, output_fn)
            elif command == ":grounding":
                if len(parts) == 1:
                    _render_grounding(runtime, output_fn)
                elif len(parts) == 2 and parts[1].lower() == "off":
                    runtime.set_grounding(None)
                    output_fn("Grounding is disabled for future turns.")
                elif len(parts) == 2 and parts[1].lower() == "reload":
                    if grounding_file is None or repository_root is None:
                        output_fn("No startup grounding file is configured.")
                    else:
                        packet = load_conversation_grounding_packet(
                            grounding_file,
                            policy=runtime.policy,
                            repository_root=repository_root,
                        )
                        runtime.set_grounding(packet)
                        output_fn("The prebuilt grounding packet was reloaded.")
                else:
                    output_fn("Usage: :grounding [off|reload]")
            elif command == ":cancel" and len(parts) == 1:
                output_fn(
                    "The nonterminal turn was cancelled."
                    if runtime.cancel()
                    else "No nonterminal turn was available to cancel."
                )
            elif command == ":resume" and len(parts) == 1:
                try:
                    _render_turn(runtime, output_fn, runtime.resume())
                except ConversationCliValidationError as exc:
                    codes = ", ".join(exc.issue_codes) or "unspecified"
                    output_fn(f"Response rejected [{exc.code}]: {codes}.")
                except ConversationCliInterruptedError as exc:
                    output_fn(f"Generation interrupted again [{exc.code}].")
                except ConversationCliCancelledError as exc:
                    output_fn(f"Generation cancelled [{exc.code}].")
                except ConversationCliTurnError as exc:
                    output_fn(f"Generation failed [{exc.code}].")
            elif command == ":close" and len(parts) == 1:
                summary = runtime.close_session()
                action = "purged" if summary.purged else "retained"
                output_fn(
                    f"Session closed and {action}: turns={summary.turn_count}; "
                    f"messages={summary.message_count}."
                )
            elif command == ":new" and len(parts) in {1, 2}:
                retention = parts[1].lower() if len(parts) == 2 else None
                runtime.new_session(retention)
                output_fn(f"New {retention or runtime.policy.default_retention} session started.")
            elif command == ":exit" and len(parts) == 1:
                if runtime.has_session:
                    summary = runtime.close_session()
                    output_fn(
                        "Session closed and "
                        + ("purged." if summary.purged else "retained.")
                    )
                output_fn("A.L.I.C.E. local runtime stopped.")
                return 0
            else:
                output_fn("Unknown command. Type :help.")
        except ConversationCliError as exc:
            output_fn(f"Runtime error: {exc}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        runtime = build_local_conversation_runtime(
            repository_root=args.repository_root,
            vault_root=args.vault_root,
            provider=args.provider,
            model=args.model,
            retention=args.retention,
            grounding_file=args.grounding_file,
        )
        return run_interactive_cli(
            runtime,
            repository_root=args.repository_root,
            grounding_file=args.grounding_file,
        )
    except (ConversationCliError, ConversationContractError, OSError) as exc:
        print(f"Unable to start A.L.I.C.E.: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
