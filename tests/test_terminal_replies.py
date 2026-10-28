"""Old clients must fail explicitly, without any ticket creation or HTTP I/O."""
import asyncio
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from hivemake_client import AsyncHiveMakeClient, HiveMakeClient, FileTicketRequest, HiveMakeConfigError
from tests.test_client import BASE


def test_return_address_rejected_locally_even_with_other_capability_flags():
    async def run():
        request = FileTicketRequest(uuid4(), "task", "Question", "Explain",
                                    reply_attachment_id=str(uuid4()), requires_human=True)
        sync = HiveMakeClient(api_key="test", base_url=BASE)
        sync._request = Mock()
        async with AsyncHiveMakeClient(api_key="test", base_url=BASE) as client:
            client._request = AsyncMock()
            with pytest.raises(HiveMakeConfigError, match="no ticket was filed"):
                sync.file_ticket(request)
            with pytest.raises(HiveMakeConfigError, match="no ticket was filed"):
                await client.file_ticket(request)
            client._request.assert_not_awaited()
            sync._request.assert_not_called()
        sync._session.close()
    asyncio.run(run())


@pytest.mark.parametrize("method", ["attach_terminal", "get_terminal_attachment", "detach_terminal"])
def test_retired_methods_do_not_contact_server(method):
    async def run():
        sync = HiveMakeClient(api_key="test", base_url=BASE)
        sync._request = Mock()
        async with AsyncHiveMakeClient(api_key="test", base_url=BASE) as client:
            client._request = AsyncMock()
            with pytest.raises(HiveMakeConfigError, match="retired"):
                getattr(sync, method)(str(uuid4()))
            with pytest.raises(HiveMakeConfigError, match="retired"):
                await getattr(client, method)(str(uuid4()))
            client._request.assert_not_awaited()
            sync._request.assert_not_called()
        sync._session.close()
    asyncio.run(run())
