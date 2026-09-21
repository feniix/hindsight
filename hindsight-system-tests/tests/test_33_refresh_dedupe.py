"""Repeated requests share a pending refresh, not a processing one.

A mental-model refresh is expensive — it drives the whole reflect loop — and the
requests arrive from several directions at once: a scheduled sweep, a
consolidation that finished, an impatient caller. Without deduplication a busy
bank queues one refresh per trigger and the worker spends its time recomputing
the same answer, which is how a slow-draining bank accumulated ~45 copies of one
model (#3487).

Explicit requests may legitimately queue a successor once the worker starts
reading: a running refresh may predate the caller's edit. These tests hold a
worker busy so the pending-state contract does not depend on machine timing.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from pathlib import Path

import pytest
from hindsight_client import Hindsight

from hindsight_system_tests import LLMStub, reflect_loop, start_hindsight_server
from hindsight_system_tests.payloads import consolidation, extracted, fact
from hindsight_system_tests.server import HindsightServer, StubServer, free_port

pytestmark = pytest.mark.asyncio

ANSWER = "Alice lives in Berlin."


@pytest.fixture(scope="module")
def hindsight_server(stub_server: StubServer, tmp_path_factory: pytest.TempPathFactory) -> Iterator[HindsightServer]:
    """One live worker slot, isolated from other suite servers' workers."""
    log_path: Path = tmp_path_factory.mktemp("refresh-dedupe-server") / "server.log"
    server = start_hindsight_server(
        stub_url=stub_server.url,
        log_path=log_path,
        database_url=f"pg0://hindsight-systest-refresh-dedupe:{free_port()}",
        extra_env={
            "HINDSIGHT_API_WORKER_MAX_SLOTS": "1",
            "HINDSIGHT_API_WORKER_CONSOLIDATION_RESERVED_SLOTS": "0",
        },
    )
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
async def model(client, llm, bank_id, settled) -> str:
    llm.on_step("extract_facts").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").returns(consolidation())
    reflect_loop(llm, answer=ANSWER)

    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.")
    await settled(bank_id)

    created = await client.mental_models.create_mental_model(
        bank_id, {"name": "Housing", "source_query": "Where does Alice live?"}
    )
    await settled(bank_id)
    return created.mental_model_id


@pytest.fixture
async def busy_worker(
    client: Hindsight,
    llm: LLMStub,
    bank_id: str,
    model: str,
    settled: Callable[[str], Awaitable[None]],
) -> AsyncIterator[str]:
    # Explicit refreshes dedupe pending work, not processing work. Hold one
    # processing refresh so the next request stays pending until assertions end.
    with llm.hold_responses() as gate:
        running = await client.mental_models.refresh_mental_model(bank_id, model)
        try:
            await gate.wait_until_entered()
            yield running.operation_id
        finally:
            gate.release()
            await settled(bank_id)


async def test_two_pending_requests_share_one_operation(
    client: Hindsight, bank_id: str, model: str, busy_worker: str
) -> None:
    """The second queued request joins the first; neither joins processing work."""
    first = await client.mental_models.refresh_mental_model(bank_id, model)
    second = await client.mental_models.refresh_mental_model(bank_id, model)

    assert first.operation_id == second.operation_id
    assert first.operation_id != busy_worker


async def test_the_bank_does_not_accumulate_a_refresh_per_request(
    client: Hindsight, bank_id: str, model: str, busy_worker: str
) -> None:
    """The #3487 shape. Five requests, and the queue must not grow five deep."""
    ids = []
    for _ in range(5):
        response = await client.mental_models.refresh_mental_model(bank_id, model)
        ids.append(response.operation_id)
    operations = await client.operations.list_operations(bank_id, type="refresh_mental_model", limit=100)

    assert len(set(ids)) == 1
    assert ids[0] != busy_worker
    assert {op.id for op in operations.operations if op.status == "pending"} == {ids[0]}
    assert {op.id for op in operations.operations if op.status == "processing"} == {busy_worker}
    # Create's completed refresh + deliberately held refresh + one queued successor.
    assert operations.total == 3


async def test_the_model_still_ends_up_refreshed(client, bank_id, model, settled):
    """Deduplication must not swallow the work — collapsing to zero refreshes
    would look identical from the queue's point of view."""
    for _ in range(3):
        await client.mental_models.refresh_mental_model(bank_id, model)
    await settled(bank_id)

    current = await client.mental_models.get_mental_model(bank_id, model, detail="full")
    assert current.content.strip() == ANSWER
    assert current.is_stale is False
