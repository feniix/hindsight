"""The provider gate must release waiters even when a story fails."""

from __future__ import annotations

import asyncio

import pytest

from hindsight_system_tests import LLMStub

pytestmark = pytest.mark.asyncio


async def test_gate_holds_and_releases_across_event_loops() -> None:
    llm = LLMStub()
    with llm.hold_responses() as gate:
        waiter = asyncio.create_task(asyncio.to_thread(asyncio.run, llm.wait_if_held()))
        await gate.wait_until_entered()
        assert not waiter.done()
        gate.release()
        gate.release()  # Both fixture cleanup and context exit can release it.
        await asyncio.wait_for(waiter, timeout=5)
    await llm.wait_if_held()


async def test_failing_story_releases_the_response_and_allows_another_gate() -> None:
    llm = LLMStub()
    with pytest.raises(ValueError, match="story failed"):
        with llm.hold_responses() as gate:
            waiter = asyncio.create_task(llm.wait_if_held())
            await gate.wait_until_entered()
            raise ValueError("story failed")
    await asyncio.wait_for(waiter, timeout=5)
    with llm.hold_responses() as gate:
        waiter = asyncio.create_task(llm.wait_if_held())
        await gate.wait_until_entered()
        assert not waiter.done()
    await asyncio.wait_for(waiter, timeout=5)


async def test_nested_gate_cannot_replace_the_active_gate() -> None:
    llm = LLMStub()
    with llm.hold_responses() as gate:
        with pytest.raises(RuntimeError, match="already active"):
            with llm.hold_responses():
                pytest.fail("Nested gate was accepted")
        waiter = asyncio.create_task(llm.wait_if_held())
        await gate.wait_until_entered()
        assert not waiter.done()
    await asyncio.wait_for(waiter, timeout=5)
