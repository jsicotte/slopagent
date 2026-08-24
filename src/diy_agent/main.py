import argparse
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ollama import chat
from ollama._types import Tool

# ── tool implementations ──────────────────────────────────────────────────────


@dataclass
class ToolFunctionAndMetadata:
    function_object: Callable[[str], str]
    ollama_definition: Tool


def read_file(path_string: str) -> str:
    p = Path(path_string)
    return p.read_text()

def list_files(path_string: str ) -> str:
    p = Path(path_string).glob("*")
    files = [str(entry) for entry in p if entry.is_file()]
    return ",".join(files)

# Small/local tool-calling models sometimes double-escape control characters
# in JSON string arguments (e.g. emit the literal two-character sequence
# "\n" instead of a real newline). Fold the common whitespace escapes back
# into real characters; anything else is left untouched.
_LITERAL_ESCAPES = {"n": "\n", "t": "\t", "r": "\r"}


def _unescape_literal_whitespace(text: str) -> str:
    return re.sub(r"\\([ntr])", lambda m: _LITERAL_ESCAPES[m.group(1)], text)

def edit_file(path_string: str, old_text: str, new_text: str) -> None:
    target_file = Path(path_string)

    old_text = _unescape_literal_whitespace(old_text)
    new_text = _unescape_literal_whitespace(new_text)

    if not target_file.exists():
        target_file.write_text(new_text)
        return

    current_content = target_file.read_text()
    new_content = current_content.replace(old_text, new_text)

    target_file.write_text(new_content)

TOOL_REGISTRY: dict[str, ToolFunctionAndMetadata] = {
    "read_file": ToolFunctionAndMetadata(
        function_object=read_file,
        ollama_definition=Tool(
            type="function",
            function=Tool.Function(
                name="read_file",
                description="Read the contents of a file",
                parameters=Tool.Function.Parameters(  # pyright: ignore[reportCallIssue] — ollama's `defs` field uses `Field(None, alias="$defs")`; pyright only infers optionality from `Field(default=...)`, misreporting `$defs` as required.
                    type="object",
                    required=["path_string"],
                    properties={
                        "path_string": Tool.Function.Parameters.Property(
                            type="string", description="The fully qualified path to a file on disk"
                        )
                    },
                ),
            ),
        ),
    ),
    "list_files": ToolFunctionAndMetadata(
        function_object=list_files,
        ollama_definition=Tool(
            type="function",
            function=Tool.Function(
                name="list_files",
                description="List the files in a directory",
                parameters=Tool.Function.Parameters(  # pyright: ignore[reportCallIssue] — ollama's `defs` field uses `Field(None, alias="$defs")`; pyright only infers optionality from `Field(default=...)`, misreporting `$defs` as required.
                    type="object",
                    required=["path_string"],
                    properties={
                        "path_string": Tool.Function.Parameters.Property(
                            type="string", description="The fully qualified path to a directory on disk"
                        )
                    },
                ),
            ),
        ),
    ),
    "edit_file": ToolFunctionAndMetadata(
        function_object=edit_file,
        ollama_definition=Tool(
            type="function",
            function=Tool.Function(
                name="edit_file",
                description="""
                    Make changes to a text file.
                    Replaces "old_str" with "new_str" in a given file. "old_str" and "new_str" must be different from each other.
                    If the file does not exist, it will be created and will contain "new_str".
                    """,
                parameters=Tool.Function.Parameters(  # pyright: ignore[reportCallIssue] — ollama's `defs` field uses `Field(None, alias="$defs")`; pyright only infers optionality from `Field(default=...)`, misreporting `$defs` as required.
                    type="object",
                    required=["path_string", "old_text", "new_text"],
                    properties={
                        "path_string": Tool.Function.Parameters.Property(
                            type="string", description="The fully qualified path to a directory on disk"
                        )
                    },
                ),
            ),
        ),
    ),
}

OOLAMA_TOOLS = [function_and_metadata.ollama_definition for function_and_metadata in TOOL_REGISTRY.values()]


# ── agent turn ────────────────────────────────────────────────────────────────


def run_agent_turn(messages: list) -> str:
    while True:
        response = chat(
            model="gemma4:e4b-mlx",
            messages=messages,
            tools=OOLAMA_TOOLS,
        )
        assistant_msg = response["message"]
        messages.append(assistant_msg)

        tool_calls = assistant_msg.get("tool_calls") or []
        if not tool_calls:
            # No more tool calls — model produced its final answer.
            return assistant_msg["content"]

        # Execute every requested tool and feed results back.
        for call in tool_calls:
            fn_name = call["function"]["name"]
            fn_args = call["function"]["arguments"]  # dict

            fn = TOOL_REGISTRY.get(fn_name).function_object
            if fn is None:
                result = f"Error: unknown tool '{fn_name}'"
            else:
                result = fn(**fn_args)

            messages.append({"role": "tool", "content": str(result)})

        # Loop: re-call the model so it can incorporate the tool results.


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="diy-agent")
    parser.add_argument(
        "-p",
        "--prompt",
        help="Run this single prompt, print the result, and exit (no interactive loop).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.prompt:
        messages = [{"role": "user", "content": args.prompt}]
        print(run_agent_turn(messages))
    else:
        messages = []

        while True:
            print("You: ", end="", flush=True)
            user_input = input()
            if not user_input.strip():
                continue

            messages.append({"role": "user", "content": user_input})

            reply = run_agent_turn(messages)

            print(f"\nLLM: {reply}\n")
