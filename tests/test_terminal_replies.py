import asyncio
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from hivemake_client import AsyncHiveMakeClient, HiveMakeClient, FileTicketRequest, HiveMakeConfigError
from tests.test_client import _ticket_payload, BASE


@pytest.mark.parametrize('supported', [False, True])
def test_return_address_capability_guard_and_payload(supported):
    async def run():
        attachment = str(uuid4())
        request = FileTicketRequest(uuid4(), 'task', 'Question', 'Explain', reply_attachment_id=attachment)
        payload = {'ticket': _ticket_payload(ticket_id=uuid4()), 'waiting_on_autonomous': True}
        responses = [{'capabilities': ['terminal_replies'] if supported else []}, payload]
        sync = HiveMakeClient(api_key='test', base_url=BASE)
        sync._request = Mock(side_effect=responses)
        async with AsyncHiveMakeClient(api_key='test', base_url=BASE) as client:
            client._request = AsyncMock(side_effect=responses)
            if supported:
                sync.file_ticket(request)
                await client.file_ticket(request)
                assert sync._request.call_args.kwargs['json_body']['reply_attachment_id'] == attachment
            else:
                with pytest.raises(HiveMakeConfigError, match='no ticket was filed'):
                    sync.file_ticket(request)
                with pytest.raises(HiveMakeConfigError, match='no ticket was filed'):
                    await client.file_ticket(request)
                assert sync._request.call_count == 1
            assert client._request.await_args_list == sync._request.call_args_list
        sync._session.close()
    asyncio.run(run())
