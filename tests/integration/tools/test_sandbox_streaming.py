#!/usr/bin/env python3
"""
Integration test for sandbox bash streaming.

Creates a real sandbox on HF, runs commands, verifies that:
  1. bash_stream() yields lines incrementally
  2. The streaming tool handler emits tool_log events
  3. The final output is collected correctly
  4. Error/exit codes propagate

Requires HF_TOKEN in environment.
"""
import asyncio
import os
import sys

sys.path.insert(0, ".")

from agent.tools.sandbox_client import Sandbox

GREEN = "\033[92m"
RED = "\033[91m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def ok(msg):
    print(f"{GREEN}  OK{RESET} {msg}")


def fail(msg):
    print(f"{RED}FAIL{RESET} {msg}")


def info(msg):
    print(f"{BLUE}INFO{RESET} {msg}")


async def test_bash_stream_basic(sb: Sandbox):
    """Test that bash_stream yields output lines and an exit event."""
    info("bash_stream: echo test")
    lines = []
    exit_code = None
    for event_type, data in sb.bash_stream("echo hello && echo world"):
        if event_type == "output":
            lines.append(data.rstrip("\n"))
        elif event_type == "exit":
            exit_code = data

    if lines == ["hello", "world"]:
        ok(f"Got expected lines: {lines}")
    else:
        fail(f"Unexpected lines: {lines}")
        return False

    if exit_code == 0:
        ok(f"Exit code: {exit_code}")
    else:
        fail(f"Unexpected exit code: {exit_code}")
        return False

    return True


async def test_bash_stream_multiline(sb: Sandbox):
    """Test streaming a script that prints multiple lines with delays."""
    info("bash_stream: multi-line with sleep")
    lines = []
    for event_type, data in sb.bash_stream(
        'for i in 1 2 3; do echo "line $i"; sleep 0.2; done'
    ):
        if event_type == "output":
            line = data.rstrip("\n")
            lines.append(line)
            info(f"  streamed: {line}")

    if len(lines) == 3 and lines == ["line 1", "line 2", "line 3"]:
        ok(f"Got all 3 lines incrementally")
    else:
        fail(f"Unexpected lines: {lines}")
        return False

    return True


async def test_bash_stream_stderr(sb: Sandbox):
    """Test that stderr is also streamed (merged into stdout)."""
    info("bash_stream: stderr output")
    lines = []
    for event_type, data in sb.bash_stream("echo out && echo err >&2"):
        if event_type == "output":
            lines.append(data.rstrip("\n"))

    if "out" in lines and "err" in lines:
        ok(f"Both stdout and stderr captured: {lines}")
    else:
        fail(f"Missing output: {lines}")
        return False

    return True


async def test_bash_stream_exit_code(sb: Sandbox):
    """Test that non-zero exit codes are reported."""
    info("bash_stream: non-zero exit code")
    exit_code = None
    for event_type, data in sb.bash_stream("exit 42"):
        if event_type == "exit":
            exit_code = data

    if exit_code == 42:
        ok(f"Got expected exit code: {exit_code}")
    else:
        fail(f"Unexpected exit code: {exit_code}")
        return False

    return True


async def test_bash_stream_empty(sb: Sandbox):
    """Test command with no output."""
    info("bash_stream: no output command")
    lines = []
    exit_code = None
    for event_type, data in sb.bash_stream("true"):
        if event_type == "output":
            lines.append(data)
        elif event_type == "exit":
            exit_code = data

    if exit_code == 0:
        ok(f"Exit code 0, output lines: {len(lines)}")
    else:
        fail(f"Unexpected exit code: {exit_code}")
        return False

    return True


async def test_bash_stream_tool_handler(sb: Sandbox):
    """Test the full tool handler path with mock session that captures events."""
    info("Tool handler: streaming bash with event capture")

    from agent.core.session import Event
    from agent.tools.sandbox_tool import _make_bash_streaming_handler

    # Mock session with event capture
    class MockSession:
        def __init__(self, sandbox):
            self.sandbox = sandbox
            self.event_queue = asyncio.Queue()
            self.events = []

        async def send_event(self, event: Event):
            self.events.append(event)
            await self.event_queue.put(event)

    session = MockSession(sb)
    handler = _make_bash_streaming_handler()

    output, success = await handler(
        {"command": 'for i in a b c; do echo "$i"; sleep 0.1; done'},
        session=session,
    )

    # Check tool_log events were emitted
    log_events = [e for e in session.events if e.event_type == "tool_log"]
    log_lines = [e.data["log"] for e in log_events]

    if log_lines == ["a", "b", "c"]:
        ok(f"Tool handler emitted {len(log_events)} tool_log events: {log_lines}")
    else:
        fail(f"Unexpected log events: {log_lines}")
        return False

    if success and "a" in output and "b" in output and "c" in output:
        ok(f"Handler returned success with full output")
    else:
        fail(f"Handler returned success={success}, output={output!r}")
        return False

    return True


async def test_original_bash_still_works(sb: Sandbox):
    """Verify the non-streaming bash endpoint still works."""
    info("Original bash endpoint (non-streaming)")
    result = await asyncio.to_thread(sb.bash, "echo legacy_test")
    if result.success and "legacy_test" in result.output:
        ok(f"Original bash works: {result.output.strip()}")
    else:
        fail(f"Original bash failed: {result}")
        return False

    return True


async def main():
    print("=" * 60)
    print(f"{BLUE}Sandbox Bash Streaming — Integration Tests{RESET}")
    print("=" * 60)

    token = os.environ.get("HF_TOKEN")
    if not token:
        fail("HF_TOKEN not set")
        sys.exit(1)

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    owner = api.whoami().get("name", "")
    info(f"HF user: {owner}")

    sb = None
    try:
        info("Creating sandbox (this takes ~2-4 minutes)...")
        sb = Sandbox.create(owner=owner, token=token, hardware="cpu-basic")
        ok(f"Sandbox ready: {sb.space_id}")
        print()

        tests = [
            test_bash_stream_basic,
            test_bash_stream_multiline,
            test_bash_stream_stderr,
            test_bash_stream_exit_code,
            test_bash_stream_empty,
            test_bash_stream_tool_handler,
            test_original_bash_still_works,
        ]

        passed = 0
        failed = 0
        for test in tests:
            print()
            try:
                result = await test(sb)
                if result:
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                fail(f"{test.__name__}: {e}")
                import traceback
                traceback.print_exc()
                failed += 1

        print()
        print("=" * 60)
        if failed == 0:
            print(f"{GREEN}All {passed} tests passed!{RESET}")
        else:
            print(f"{RED}{failed} failed{RESET}, {GREEN}{passed} passed{RESET}")
        print("=" * 60)

    finally:
        if sb and sb._owns_space:
            info(f"Cleaning up sandbox: {sb.space_id}")
            try:
                sb.delete()
                ok("Sandbox deleted")
            except Exception as e:
                fail(f"Cleanup failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
