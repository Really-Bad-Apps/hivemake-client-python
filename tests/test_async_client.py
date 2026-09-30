"""Async contracts and real HTTP cancellation, timeout and pool behavior."""
import asyncio
from contextlib import asynccontextmanager
import inspect
import json
from unittest.mock import AsyncMock, Mock
from uuid import uuid4
import httpx
import pytest
from hivemake_client import AsyncHiveMakeClient, HiveMakeClient, FileTicketRequest
from hivemake_client.exceptions import (HiveMakeAPIError, HiveMakeAuthError,
    HiveMakeForbidden, HiveMakeNotFound, HiveMakeConflict, HiveMakeValidationError,
    HiveMakeServerError, HiveMakeConfigError)
from tests.test_client import _ticket_payload, BASE

TICKET_ID, PROJECT_ID = str(uuid4()), str(uuid4())
TICKET = _ticket_payload(ticket_id=uuid4())
AGENT = dict(id=str(uuid4()), hive_id=str(uuid4()), project_id=PROJECT_ID,
             name='test', description='test agent', status='active',
             created_at=1700000000, updated_at=1700000000)
OUTBOUND = dict(ticket=TICKET, waiting_on_autonomous=False, is_scheduled=True)
CASES = [
    ('file_ticket', (FileTicketRequest(PROJECT_ID, 'task', 'Title', 'Description'),), {}, OUTBOUND),
    ('get_ticket', (TICKET_ID,), {}, dict(ticket=TICKET, negotiations=[], history=[], waiting_on='assignee')),
    ('check_tickets', (), {'scheduled_offset': 15}, dict(inbox=[TICKET], scheduled=[TICKET], count=2)),
    ('list_inbox', (), {'status': 'open', 'include_terminal': True, 'q': 'title'}, dict(tickets=[TICKET])),
    ('list_outbox', (), {'q': 'title'}, dict(tickets=[OUTBOUND])),
    *[(name, (TICKET_ID, 'reason'), {}, dict(ticket=TICKET)) for name in
      ('accept', 'reject', 'resolve', 'close', 'withdraw', 'cancel_info_request', 'provide_info', 'add_note', 'escalate')],
    *[(name, (TICKET_ID, 'reason'), {}, OUTBOUND) for name in ('reopen', 'request_info')],
    ('reschedule', (TICKET_ID, None), {}, OUTBOUND),
    ('redirect', (TICKET_ID, PROJECT_ID, 'reason'), {}, OUTBOUND),
    ('register', ('my capabilities',), {}, dict(agent=AGENT)),
    ('me', (), {}, dict(agent=AGENT)),
    ('discover_agents', ('query',), {'limit': 4, 'min_score': .3}, dict(matches=[])),
    ('find_similar_tickets', ('query',), {'ticket_type': 'bug', 'limit': 4}, []),
    ('recall_knowledge', ('query',), {}, dict(answer='answer')),
    ('admin_usage', (), {}, dict(run_id=str(uuid4()), method_version='v1', owners=[])),
    ('add_learning', ('content',), {'category': 'test', 'source_ticket_id': TICKET_ID}, dict(learning_id=str(uuid4()))),
]

@pytest.mark.parametrize('method,args,kwargs,payload', CASES, ids=[c[0] for c in CASES])
def test_endpoint_contract_matches_sync(method, args, kwargs, payload):
    async def run():
        sync = HiveMakeClient(api_key='test', base_url=BASE)
        sync._request = Mock(return_value=payload)
        async with AsyncHiveMakeClient(api_key='test', base_url=BASE) as client:
            client._request = AsyncMock(return_value=payload)
            assert inspect.signature(getattr(client, method)) == inspect.signature(getattr(sync, method))
            assert await getattr(client, method)(*args, **kwargs) == getattr(sync, method)(*args, **kwargs)
            assert client._request.await_args_list == sync._request.call_args_list
        sync._session.close()
    asyncio.run(run())


def test_all_sync_endpoints_are_covered():
    assert {name for name, fn in inspect.getmembers(HiveMakeClient, inspect.isfunction)
            if not name.startswith('_')} == {case[0] for case in CASES}


@pytest.mark.parametrize('status,error', [(401,HiveMakeAuthError), (403,HiveMakeForbidden),
    (404,HiveMakeNotFound), (409,HiveMakeConflict), (400,HiveMakeValidationError),
    (422,HiveMakeValidationError), (429,HiveMakeAPIError), (500,HiveMakeServerError), (503,HiveMakeServerError)])
def test_api_error_types_and_details(status, error):
    async def run():
        calls = []
        async def handler(request):
            calls.append(request)
            return httpx.Response(status, json={'error':'test_code', 'detail':'test detail'})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = AsyncHiveMakeClient(api_key='test', http_client=http)
            with pytest.raises(error) as exc: await client.add_note(TICKET_ID, 'write once')
            assert (exc.value.status_code, exc.value.error_code, exc.value.detail) == (status,'test_code','test detail')
            assert len(calls) == 1
    asyncio.run(run())


def test_non_json_error_and_client_ownership():
    async def run():
        async def handler(request): return httpx.Response(502, text='bad gateway')
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            async with AsyncHiveMakeClient(api_key='test', http_client=http) as client:
                with pytest.raises(HiveMakeServerError, match='Bad Gateway'): await client.check_tickets()
            assert not http.is_closed
            with pytest.raises(RuntimeError, match='closed'): await client.check_tickets()
        assert http.is_closed
        async with AsyncHiveMakeClient(api_key='test') as owned: pass
        assert owned._http_client.is_closed
        await owned.aclose()
    asyncio.run(run())


def test_request_identity_and_timeout_are_not_shared():
    async def run():
        seen = []
        async def handler(request):
            seen.append(request)
            await asyncio.sleep(.001)
            return httpx.Response(200, json={'answer':'ok'}, headers={'Set-Cookie':'identity=other'})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler),
                auth=('wrong','credentials'), cookies={'identity':'wrong'},
                headers={'Authorization':'wrong'}, params={'wrong':'user'}) as http:
            clients = [AsyncHiveMakeClient(api_key=f'caller-{i}', http_client=http) for i in range(10)]
            await asyncio.gather(*(c.recall_knowledge('query') for c in clients))
            await clients[0].check_tickets()
            assert {r.headers['Authorization'] for r in seen[:10]} == {f'Bearer caller-{i}' for i in range(10)}
            assert all('cookie' not in r.headers and not r.url.query for r in seen)
            assert all(r.extensions['timeout'] == dict(connect=120.,read=120.,write=120.,pool=5.) for r in seen[:10])
            assert seen[-1].extensions['timeout']['read'] == 30.
            assert http.headers['Authorization'] == 'wrong'
    asyncio.run(run())


def test_scheduling_capability_guard_and_validation():
    async def run():
        seen, supports = [], False
        async def handler(request):
            seen.append(request)
            if request.url.path == '/api/health':
                return httpx.Response(200,json={'capabilities':['scheduled_tickets'] if supports else []})
            return httpx.Response(201,json=OUTBOUND)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = AsyncHiveMakeClient(api_key='test',http_client=http)
            req = FileTicketRequest(PROJECT_ID,'task','Later','D',not_before=1790499600)
            with pytest.raises(HiveMakeConfigError,match='no ticket was filed'): await client.file_ticket(req)
            assert len(seen) == 1
            supports = True
            assert (await client.file_ticket(req)).is_scheduled
            assert json.loads(seen[-1].content)['not_before'] == 1790499600
            for offset in (-1,True,'0'):
                with pytest.raises(ValueError): await client.check_tickets(scheduled_offset=offset)
            assert len(seen) == 3
    asyncio.run(run())


@asynccontextmanager
async def slow_backend():
    tasks, calls = set(), []
    started, release = asyncio.Event(), asyncio.Event()
    async def handler(reader, writer):
        tasks.add(asyncio.current_task())
        try:
            calls.append(await reader.readuntil(b'\r\n\r\n'))
            started.set()
            await release.wait()
            writer.write(b'HTTP/1.1 200 OK\r\nContent-Length: 11\r\nConnection: close\r\n\r\n{"count":0}')
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            tasks.discard(asyncio.current_task())
    listener = await asyncio.start_server(handler,'127.0.0.1',0)
    try:
        yield f'http://127.0.0.1:{listener.sockets[0].getsockname()[1]}',started,release,calls
    finally:
        listener.close()
        for task in list(tasks): task.cancel()
        await asyncio.gather(*list(tasks),return_exceptions=True)
        await listener.wait_closed()


def test_pool_wait_timeout_cancellation_and_recovery():
    async def run():
        async with slow_backend() as (url,started,release,calls):
            async with httpx.AsyncClient(limits=httpx.Limits(max_connections=1),trust_env=False) as http:
                client = AsyncHiveMakeClient(api_key='test',base_url=url,http_client=http,pool_timeout=.02)
                first = asyncio.create_task(client.check_tickets())
                await asyncio.wait_for(started.wait(),1)
                with pytest.raises(httpx.PoolTimeout): await client.check_tickets()
                assert len(calls) == 1
                first.cancel()
                with pytest.raises(asyncio.CancelledError): await first
                release.set()
                assert (await client.check_tickets()).count == 0
                assert len(calls) == 2
    asyncio.run(run())


def test_real_read_timeout_is_not_retried():
    async def run():
        async with slow_backend() as (url,started,release,calls):
            async with AsyncHiveMakeClient(api_key='test',base_url=url,timeout=.02) as client:
                with pytest.raises(httpx.ReadTimeout): await client.add_note(TICKET_ID,'one write')
                assert len(calls) == 1
    asyncio.run(run())


def test_redirect_is_not_followed_and_write_is_not_replayed():
    async def run():
        seen=[]
        async def handler(request):
            seen.append(request)
            return httpx.Response(307,headers={'Location':'https://other.example/leak'})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler),follow_redirects=True) as http:
            client=AsyncHiveMakeClient(api_key='private',http_client=http)
            with pytest.raises(HiveMakeAPIError) as exc: await client.add_note(TICKET_ID,'write')
            assert exc.value.status_code == 307 and len(seen) == 1
    asyncio.run(run())


def test_async_configuration_matches_sync(monkeypatch):
    monkeypatch.delenv('HIVEMAKE_API_KEY',raising=False)
    with pytest.raises(HiveMakeConfigError): AsyncHiveMakeClient()
    monkeypatch.setenv('HIVEMAKE_API_KEY','from-env')
    monkeypatch.setenv('HIVEMAKE_API_URL','http://localhost:5001/')
    async def run():
        async with AsyncHiveMakeClient() as client:
            assert client.api_key == 'from-env' and client.base_url == 'http://localhost:5001'
        async with AsyncHiveMakeClient(api_key='explicit',base_url=BASE) as client:
            assert client.api_key == 'explicit' and client.base_url == BASE
    asyncio.run(run())
