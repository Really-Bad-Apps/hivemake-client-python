import json
from uuid import uuid4

import responses

import pytest
from hivemake_client import FileTicketRequest, HiveMakeClient, HiveMakeConfigError
from tests.test_client import _ticket_payload, BASE


@responses.activate
def test_schedule_roundtrip_and_reschedule_payload():
    ticket = _ticket_payload(ticket_id=uuid4())
    ticket['not_before'] = 1790499600
    result = {'ticket': ticket, 'waiting_on_autonomous': False, 'is_scheduled': True}
    responses.get(BASE + '/api/health', json={'capabilities': ['scheduled_tickets']})
    responses.post(BASE + '/api/tickets', json=result, status=201)
    client = HiveMakeClient(api_key='hm_test_abc', base_url=BASE)
    filed = client.file_ticket(FileTicketRequest(uuid4(), 'task', 'Later', 'D', not_before=1790499600))
    assert filed.is_scheduled and filed.ticket.not_before == 1790499600
    assert json.loads(responses.calls[1].request.body)['not_before'] == 1790499600
    responses.get(BASE + '/api/tickets/check', json={'scheduled': [ticket], 'scheduled_truncated': True, 'count': 1})
    queue = client.check_tickets(scheduled_offset=15)
    assert queue.scheduled[0].not_before == 1790499600 and queue.scheduled_truncated
    assert responses.calls[-1].request.url.endswith('scheduled_offset=15')
    responses.post(BASE + '/api/tickets/' + ticket['id'] + '/negotiations', json=result, status=201)
    client.reschedule(ticket['id'], None, 'Release')
    assert json.loads(responses.calls[-1].request.body) == {'action': 'rescheduled', 'not_before': None, 'message': 'Release'}


@responses.activate
def test_old_server_cannot_silently_file_scheduled_work_immediately():
    responses.get(BASE + '/api/health', json={'status': 'ok'})
    client = HiveMakeClient(api_key='hm_test_abc', base_url=BASE)
    with pytest.raises(HiveMakeConfigError, match='no ticket was filed'):
        client.file_ticket(FileTicketRequest(uuid4(), 'task', 'Later', 'D', not_before=1790499600))
    assert len(responses.calls) == 1 and responses.calls[0].request.method == 'GET'


@responses.activate
@pytest.mark.parametrize('offset', [-1, True, 1.5, '0', 9223372036854775808])
def test_invalid_pagination_never_reaches_api(offset):
    client = HiveMakeClient(api_key='hm_test_abc', base_url=BASE)
    with pytest.raises(ValueError, match='scheduled_offset'):
        client.check_tickets(scheduled_offset=offset)
    assert not responses.calls
