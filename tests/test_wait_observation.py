"""Old servers and malformed wait metadata must not become successful waits."""

import asyncio
from uuid import uuid4

import httpx
import pytest
import responses

from hivemake_client import AsyncHiveMakeClient, HiveMakeClient, HiveMakeNotFound
from hivemake_client.client import _ticket_wait_from_payload
from tests.test_client import BASE, _ticket_payload


@pytest.mark.parametrize("status", [None, "unknown", [], True])
def test_unknown_wait_contract_fails_explicitly(status) -> None:
    ticket_id = uuid4()
    data = {"ticket": _ticket_payload(ticket_id=ticket_id), "wait_status": status}
    with responses.RequestsMock() as mock:
        mock.get(f"{BASE}/api/tickets/{ticket_id}/wait", json=data)
        with pytest.raises(ValueError, match="wait_status"):
            HiveMakeClient(api_key="test", base_url=BASE).observe_ticket(ticket_id)

    async def run() -> None:
        async def backend(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=data)
        async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as http:
            with pytest.raises(ValueError, match="wait_status"):
                await AsyncHiveMakeClient(api_key="test", http_client=http).observe_ticket(ticket_id)
    asyncio.run(run())


def test_legacy_server_does_not_fall_back_to_consuming_read() -> None:
    ticket_id = uuid4()
    with responses.RequestsMock() as mock:
        mock.get(f"{BASE}/api/tickets/{ticket_id}/wait", status=404, json={"error": "not_found"})
        with pytest.raises(HiveMakeNotFound):
            HiveMakeClient(api_key="test", base_url=BASE).observe_ticket(ticket_id)
        assert len(mock.calls) == 1


@pytest.mark.parametrize("malformed", [
    [], None, {"ticket": []},
    {"negotiations": [None]}, {"history": {}}, {"assigned_agent": "invalid"},
])
def test_malformed_wait_objects_fail_explicitly(malformed) -> None:
    data = malformed
    if isinstance(malformed, dict):
        data = {"ticket": _ticket_payload(ticket_id=uuid4()), "wait_status": "resolved", **malformed}
    with pytest.raises(ValueError, match="Invalid ticket wait response"):
        _ticket_wait_from_payload(data)
