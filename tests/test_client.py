"""Tests for hivemake_client.HiveMakeClient.

Uses the `responses` library to mock the HiveMake server. No real network.
"""

from typing import Any, Optional
from uuid import UUID, uuid4

import pytest
import responses

from hivemake_client import (
    FileTicketRequest,
    HiveMakeAPIError,
    HiveMakeAuthError,
    HiveMakeClient,
    HiveMakeConfigError,
    HiveMakeConflict,
    HiveMakeForbidden,
    HiveMakeNotFound,
    HiveMakeServerError,
    HiveMakeValidationError,
)
from hivemake_models import (
    NegotiationAction,
    TicketPriority,
    TicketStatus,
    TicketType,
)


BASE = "https://api.test.local"


def _ticket_payload(
    *,
    ticket_id: UUID,
    title: str = "T",
    status: str = "open",
    assigned_agent_id: Optional[UUID] = None,
) -> dict[str, Any]:
    """A minimal-but-valid Ticket JSON payload, shaped like server output."""
    return {
        "id": str(ticket_id),
        "hive_id": str(uuid4()),
        "project_id": str(uuid4()),
        "created_by_agent_id": str(uuid4()),
        "ticket_type": "bug",
        "title": title,
        "description": "D",
        "priority": "medium",
        "status": status,
        "created_at": 1700000000,
        "updated_at": 1700000001,
        "assigned_agent_id": str(assigned_agent_id) if assigned_agent_id else None,
        "requested_by_user_id": None,
        "resolution": None,
    }


@pytest.fixture
def client() -> HiveMakeClient:
    return HiveMakeClient(api_key="hm_test_abc", base_url=BASE)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_constructor_reads_env_var(self, monkeypatch) -> None:
        monkeypatch.setenv("HIVEMAKE_API_KEY", "hm_env_key")
        c = HiveMakeClient()
        assert c.api_key == "hm_env_key"

    def test_constructor_param_overrides_env(self, monkeypatch) -> None:
        monkeypatch.setenv("HIVEMAKE_API_KEY", "hm_env_key")
        c = HiveMakeClient(api_key="hm_explicit")
        assert c.api_key == "hm_explicit"

    def test_no_key_raises_config_error(self, monkeypatch) -> None:
        monkeypatch.delenv("HIVEMAKE_API_KEY", raising=False)
        with pytest.raises(HiveMakeConfigError):
            HiveMakeClient()

    def test_base_url_default(self, monkeypatch) -> None:
        monkeypatch.delenv("HIVEMAKE_API_URL", raising=False)
        c = HiveMakeClient(api_key="hm_test")
        assert c.base_url == "https://api.hivemake.ai"

    def test_base_url_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("HIVEMAKE_API_URL", "https://staging.example.com/")
        c = HiveMakeClient(api_key="hm_test")
        # Trailing slash stripped.
        assert c.base_url == "https://staging.example.com"

    def test_authorization_header_set(self, client) -> None:
        assert client._session.headers["Authorization"] == "Bearer hm_test_abc"


# ---------------------------------------------------------------------------
# file_ticket
# ---------------------------------------------------------------------------

class TestFileTicket:

    @responses.activate
    def test_file_ticket_happy(self, client) -> None:
        ticket_id = uuid4()
        target_id = uuid4()
        responses.post(
            f"{BASE}/api/tickets",
            json={"ticket": _ticket_payload(ticket_id=ticket_id)},
            status=201,
        )

        ticket = client.file_ticket(FileTicketRequest(
            target_project_id=target_id,
            ticket_type=TicketType.BUG,
            title="X",
            description="Y",
            priority=TicketPriority.HIGH,
        ))

        assert isinstance(ticket.id, UUID)
        assert ticket.id == ticket_id
        # Request body sent the wire-strings for enums.
        sent = responses.calls[0].request.body
        assert b'"ticket_type": "bug"' in sent
        assert b'"priority": "high"' in sent
        assert f'"target_project_id": "{target_id}"'.encode() in sent

    @responses.activate
    def test_file_ticket_accepts_string_enums(self, client) -> None:
        responses.post(
            f"{BASE}/api/tickets",
            json={"ticket": _ticket_payload(ticket_id=uuid4())},
            status=201,
        )

        client.file_ticket(FileTicketRequest(
            target_project_id=str(uuid4()),
            ticket_type="bug",
            title="X",
            description="Y",
            priority="medium",
        ))

        sent = responses.calls[0].request.body
        assert b'"ticket_type": "bug"' in sent
        assert b'"priority": "medium"' in sent

    @responses.activate
    def test_target_project_not_found(self, client) -> None:
        responses.post(
            f"{BASE}/api/tickets",
            json={"error": "target_project_not_found"},
            status=404,
        )
        with pytest.raises(HiveMakeNotFound) as exc:
            client.file_ticket(FileTicketRequest(
                target_project_id=uuid4(),
                ticket_type=TicketType.BUG,
                title="X", description="Y",
            ))
        assert exc.value.error_code == "target_project_not_found"
        assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# list_inbox
# ---------------------------------------------------------------------------

class TestListInbox:

    @responses.activate
    def test_empty_inbox(self, client) -> None:
        responses.get(f"{BASE}/api/tickets", json={"tickets": []}, status=200)
        assert client.list_inbox() == []

    @responses.activate
    def test_returns_tickets(self, client) -> None:
        ids = [uuid4(), uuid4()]
        responses.get(
            f"{BASE}/api/tickets",
            json={"tickets": [_ticket_payload(ticket_id=i) for i in ids]},
            status=200,
        )

        result = client.list_inbox()
        assert [t.id for t in result] == ids
        assert all(isinstance(t.id, UUID) for t in result)

    @responses.activate
    def test_status_filter_passed_as_query_param(self, client) -> None:
        responses.get(
            f"{BASE}/api/tickets",
            json={"tickets": []}, status=200,
        )
        client.list_inbox(status=TicketStatus.OPEN)
        assert responses.calls[0].request.url.endswith("?status=open")

    @responses.activate
    def test_status_filter_accepts_string(self, client) -> None:
        responses.get(f"{BASE}/api/tickets", json={"tickets": []}, status=200)
        client.list_inbox(status="accepted")
        assert responses.calls[0].request.url.endswith("?status=accepted")

    @responses.activate
    def test_include_terminal_flag_passed_as_query_param(self, client) -> None:
        responses.get(f"{BASE}/api/tickets", json={"tickets": []}, status=200)
        client.list_inbox(include_terminal=True)
        assert "include_terminal=true" in responses.calls[0].request.url

    @responses.activate
    def test_status_and_include_terminal_both_sent(self, client) -> None:
        """Server-side precedence handles the conflict; client just forwards both."""
        responses.get(f"{BASE}/api/tickets", json={"tickets": []}, status=200)
        client.list_inbox(status="open", include_terminal=True)
        url = responses.calls[0].request.url
        assert "status=open" in url
        assert "include_terminal=true" in url

    @responses.activate
    def test_invalid_status_raises(self, client) -> None:
        responses.get(
            f"{BASE}/api/tickets",
            json={"error": "invalid_status"}, status=400,
        )
        with pytest.raises(HiveMakeValidationError) as exc:
            client.list_inbox(status="bogus")
        assert exc.value.error_code == "invalid_status"


# ---------------------------------------------------------------------------
# get_ticket
# ---------------------------------------------------------------------------

class TestGetTicket:

    @responses.activate
    def test_returns_ticket_with_thread(self, client) -> None:
        ticket_id = uuid4()
        hive_id = uuid4()
        agent_id = uuid4()
        responses.get(
            f"{BASE}/api/tickets/{ticket_id}",
            json={
                "ticket": _ticket_payload(ticket_id=ticket_id),
                "negotiations": [
                    {
                        "id": str(uuid4()),
                        "hive_id": str(hive_id),
                        "ticket_id": str(ticket_id),
                        "action": "info_requested",
                        "message": "what version?",
                        "from_agent_id": str(agent_id),
                        "from_user_id": None,
                        "to_agent_id": None,
                        "to_user_id": None,
                        "metadata": {},
                        "created_at": 1700000005,
                    },
                ],
                "history": [
                    {
                        "id": str(uuid4()),
                        "hive_id": str(hive_id),
                        "ticket_id": str(ticket_id),
                        "field_changed": "status",
                        "old_value": "accepted",
                        "new_value": "info_requested",
                        "actor_agent_id": str(agent_id),
                        "actor_user_id": None,
                        "created_at": 1700000005,
                    },
                ],
            },
            status=200,
        )

        detail = client.get_ticket(ticket_id)
        assert detail.ticket.id == ticket_id
        assert len(detail.negotiations) == 1
        assert detail.negotiations[0].action == NegotiationAction.INFO_REQUESTED
        assert detail.negotiations[0].message == "what version?"
        assert isinstance(detail.negotiations[0].id, UUID)
        assert len(detail.history) == 1
        assert detail.history[0].field_changed == "status"
        assert detail.history[0].new_value == "info_requested"

    @responses.activate
    def test_empty_thread_ok(self, client) -> None:
        """Server may omit negotiations / history (or send empty lists);
        both should yield empty client-side lists."""
        ticket_id = uuid4()
        responses.get(
            f"{BASE}/api/tickets/{ticket_id}",
            json={"ticket": _ticket_payload(ticket_id=ticket_id)},
            status=200,
        )
        detail = client.get_ticket(ticket_id)
        assert detail.ticket.id == ticket_id
        assert detail.negotiations == []
        assert detail.history == []

    @responses.activate
    def test_not_found_raises_typed_error(self, client) -> None:
        """The dual-auth endpoint folds "unauthorized" callers into 404 to
        avoid leaking ticket existence — so this is the realistic error
        a `get_ticket` call hits when the caller isn't the creator or
        assignee and isn't a hive member."""
        ticket_id = uuid4()
        responses.get(
            f"{BASE}/api/tickets/{ticket_id}",
            json={"error": "not_found"}, status=404,
        )
        with pytest.raises(HiveMakeNotFound):
            client.get_ticket(ticket_id)


# ---------------------------------------------------------------------------
# list_outbox
# ---------------------------------------------------------------------------

class TestListOutbox:

    @responses.activate
    def test_empty_outbox(self, client) -> None:
        responses.get(f"{BASE}/api/tickets/outbox", json={"tickets": []}, status=200)
        assert client.list_outbox() == []

    @responses.activate
    def test_returns_tickets(self, client) -> None:
        ids = [uuid4(), uuid4()]
        responses.get(
            f"{BASE}/api/tickets/outbox",
            json={"tickets": [_ticket_payload(ticket_id=i) for i in ids]},
            status=200,
        )

        result = client.list_outbox()
        assert [t.id for t in result] == ids
        assert all(isinstance(t.id, UUID) for t in result)

    @responses.activate
    def test_status_filter_passed_as_query_param(self, client) -> None:
        responses.get(
            f"{BASE}/api/tickets/outbox",
            json={"tickets": []}, status=200,
        )
        client.list_outbox(status=TicketStatus.OPEN)
        assert responses.calls[0].request.url.endswith("?status=open")

    @responses.activate
    def test_status_filter_accepts_string(self, client) -> None:
        responses.get(f"{BASE}/api/tickets/outbox", json={"tickets": []}, status=200)
        client.list_outbox(status="accepted")
        assert responses.calls[0].request.url.endswith("?status=accepted")

    @responses.activate
    def test_include_terminal_flag_passed_as_query_param(self, client) -> None:
        responses.get(f"{BASE}/api/tickets/outbox", json={"tickets": []}, status=200)
        client.list_outbox(include_terminal=True)
        assert "include_terminal=true" in responses.calls[0].request.url

    @responses.activate
    def test_status_and_include_terminal_both_sent(self, client) -> None:
        responses.get(f"{BASE}/api/tickets/outbox", json={"tickets": []}, status=200)
        client.list_outbox(status="open", include_terminal=True)
        url = responses.calls[0].request.url
        assert "status=open" in url
        assert "include_terminal=true" in url


# ---------------------------------------------------------------------------
# Negotiation actions
# ---------------------------------------------------------------------------

class TestActions:

    @responses.activate
    def test_accept(self, client) -> None:
        tid = uuid4()
        responses.post(
            f"{BASE}/api/tickets/{tid}/negotiations",
            json={
                "ticket": _ticket_payload(ticket_id=tid, status="accepted"),
                "negotiation": {"id": str(uuid4()), "action": "accepted"},
            },
            status=201,
        )
        ticket = client.accept(tid, message="on it")
        assert ticket.id == tid
        body = responses.calls[0].request.body
        assert b'"action": "accepted"' in body
        assert b'"message": "on it"' in body

    @responses.activate
    def test_reject(self, client) -> None:
        tid = uuid4()
        responses.post(
            f"{BASE}/api/tickets/{tid}/negotiations",
            json={
                "ticket": _ticket_payload(ticket_id=tid, status="rejected"),
                "negotiation": {"id": str(uuid4()), "action": "rejected"},
            },
            status=201,
        )
        ticket = client.reject(tid, message="not ours")
        assert ticket.id == tid
        body = responses.calls[0].request.body
        assert b'"action": "rejected"' in body
        assert b'"message": "not ours"' in body

    def test_reject_requires_message_positionally(self, client) -> None:
        with pytest.raises(TypeError):
            client.reject(uuid4())

    @responses.activate
    def test_reject_empty_message_maps_to_validation_error(self, client) -> None:
        tid = uuid4()
        responses.post(
            f"{BASE}/api/tickets/{tid}/negotiations",
            json={"error": "empty_message", "detail": "message required"},
            status=422,
        )
        with pytest.raises(HiveMakeValidationError):
            client.reject(tid, message="")

    @responses.activate
    def test_resolve(self, client) -> None:
        tid = uuid4()
        responses.post(
            f"{BASE}/api/tickets/{tid}/negotiations",
            json={
                "ticket": _ticket_payload(ticket_id=tid, status="resolved"),
                "negotiation": {"id": str(uuid4()), "action": "resolved"},
            },
            status=201,
        )
        ticket = client.resolve(tid, message="shipped")
        assert ticket.id == tid
        body = responses.calls[0].request.body
        assert b'"action": "resolved"' in body
        assert b'"message": "shipped"' in body

    def test_resolve_requires_message_positionally(self, client) -> None:
        """`message` is required — no default — so the SDK fails at the call
        site rather than reaching the wire with an empty string."""
        with pytest.raises(TypeError):
            client.resolve(uuid4())

    @responses.activate
    def test_resolve_empty_message_maps_to_validation_error(self, client) -> None:
        """Server returns 422 for empty messages; client raises validation."""
        tid = uuid4()
        responses.post(
            f"{BASE}/api/tickets/{tid}/negotiations",
            json={"error": "validation_error", "detail": "message required"},
            status=422,
        )
        with pytest.raises(HiveMakeValidationError):
            client.resolve(tid, message="")

    @responses.activate
    def test_reopen(self, client) -> None:
        tid = uuid4()
        responses.post(
            f"{BASE}/api/tickets/{tid}/negotiations",
            json={
                "ticket": _ticket_payload(ticket_id=tid, status="open"),
                "negotiation": {"id": str(uuid4()), "action": "reopened"},
            },
            status=201,
        )
        ticket = client.reopen(tid, message="regression in panel-3")
        assert ticket.id == tid
        body = responses.calls[0].request.body
        assert b'"action": "reopened"' in body
        assert b'"message": "regression in panel-3"' in body

    def test_reopen_requires_message_positionally(self, client) -> None:
        with pytest.raises(TypeError):
            client.reopen(uuid4())

    @responses.activate
    def test_reopen_empty_message_maps_to_validation_error(self, client) -> None:
        tid = uuid4()
        responses.post(
            f"{BASE}/api/tickets/{tid}/negotiations",
            json={"error": "validation_error", "detail": "message required"},
            status=422,
        )
        with pytest.raises(HiveMakeValidationError):
            client.reopen(tid, message="")

    @responses.activate
    def test_redirect(self, client) -> None:
        tid = uuid4()
        target = uuid4()
        responses.post(
            f"{BASE}/api/tickets/{tid}/negotiations",
            json={
                "ticket": _ticket_payload(ticket_id=tid, status="open"),
                "negotiation": {"id": str(uuid4()), "action": "redirected"},
            },
            status=201,
        )
        client.redirect(tid, target_project_id=target, message="wrong team")
        body = responses.calls[0].request.body
        assert b'"action": "redirected"' in body
        assert f'"target_project_id": "{target}"'.encode() in body

    @responses.activate
    def test_request_info(self, client) -> None:
        tid = uuid4()
        responses.post(
            f"{BASE}/api/tickets/{tid}/negotiations",
            json={
                "ticket": _ticket_payload(ticket_id=tid),
                "negotiation": {"id": str(uuid4()), "action": "info_requested"},
            },
            status=201,
        )
        client.request_info(tid, message="repro steps?")
        assert b'"action": "info_requested"' in responses.calls[0].request.body

    @responses.activate
    def test_provide_info(self, client) -> None:
        tid = uuid4()
        responses.post(
            f"{BASE}/api/tickets/{tid}/negotiations",
            json={
                "ticket": _ticket_payload(ticket_id=tid),
                "negotiation": {"id": str(uuid4()), "action": "info_provided"},
            },
            status=201,
        )
        client.provide_info(tid, message="see attached")
        assert b'"action": "info_provided"' in responses.calls[0].request.body

    @responses.activate
    def test_close(self, client) -> None:
        tid = uuid4()
        responses.post(
            f"{BASE}/api/tickets/{tid}/negotiations",
            json={
                "ticket": _ticket_payload(ticket_id=tid, status="closed"),
                "negotiation": {"id": str(uuid4()), "action": "closed"},
            },
            status=201,
        )
        ticket = client.close(tid, message="dupe of #123")
        assert ticket.id == tid
        body = responses.calls[0].request.body
        assert b'"action": "closed"' in body
        assert b'"message": "dupe of #123"' in body

    def test_close_requires_message_positionally(self, client) -> None:
        with pytest.raises(TypeError):
            client.close(uuid4())

    @responses.activate
    def test_close_empty_message_maps_to_validation_error(self, client) -> None:
        tid = uuid4()
        responses.post(
            f"{BASE}/api/tickets/{tid}/negotiations",
            json={"error": "empty_message", "detail": "message required"},
            status=422,
        )
        with pytest.raises(HiveMakeValidationError):
            client.close(tid, message="")

    @responses.activate
    def test_withdraw(self, client) -> None:
        tid = uuid4()
        responses.post(
            f"{BASE}/api/tickets/{tid}/negotiations",
            json={
                "ticket": _ticket_payload(ticket_id=tid, status="withdrawn"),
                "negotiation": {"id": str(uuid4()), "action": "withdrawn"},
            },
            status=201,
        )
        ticket = client.withdraw(tid, message="never mind")
        assert ticket.id == tid
        body = responses.calls[0].request.body
        assert b'"action": "withdrawn"' in body
        assert b'"message": "never mind"' in body

    @responses.activate
    def test_escalate(self, client) -> None:
        tid = uuid4()
        responses.post(
            f"{BASE}/api/tickets/{tid}/negotiations",
            json={
                "ticket": _ticket_payload(ticket_id=tid, status="escalated"),
                "negotiation": {"id": str(uuid4()), "action": "escalated"},
            },
            status=201,
        )
        ticket = client.escalate(tid, message="discover empty")
        assert ticket.id == tid
        body = responses.calls[0].request.body
        assert b'"action": "escalated"' in body
        assert b'"message": "discover empty"' in body

    @responses.activate
    def test_invalid_transition_raises_conflict(self, client) -> None:
        tid = uuid4()
        responses.post(
            f"{BASE}/api/tickets/{tid}/negotiations",
            json={"error": "invalid_transition", "detail": "cannot accept from accepted"},
            status=409,
        )
        with pytest.raises(HiveMakeConflict) as exc:
            client.accept(tid)
        assert exc.value.error_code == "invalid_transition"
        assert exc.value.detail == "cannot accept from accepted"
        assert "cannot accept from accepted" in str(exc.value)


# ---------------------------------------------------------------------------
# Agent registration + discovery
# ---------------------------------------------------------------------------

def _agent_payload(
    *, agent_id: UUID, registered_at: Optional[int] = 1700000010,
) -> dict[str, Any]:
    return {
        "id": str(agent_id),
        "hive_id": str(uuid4()),
        "project_id": str(uuid4()),
        "name": "Test Agent",
        "description": "a test agent",
        "config": {},
        "status": "active",
        "registered_at": registered_at,
        "created_at": 1700000000,
        "updated_at": 1700000010,
    }


class TestRegister:

    @responses.activate
    def test_register_posts_description(self, client) -> None:
        agent_id = uuid4()
        responses.post(
            f"{BASE}/api/agents/register",
            json={"agent": _agent_payload(agent_id=agent_id)},
            status=200,
        )
        result = client.register("Watches loki logs and files bug tickets")

        assert result.agent.id == agent_id
        assert result.agent.registered_at == 1700000010
        assert b'"description":' in responses.calls[0].request.body

    @responses.activate
    def test_register_validation_error(self, client) -> None:
        responses.post(
            f"{BASE}/api/agents/register",
            json={"error": "invalid_description"}, status=422,
        )
        with pytest.raises(HiveMakeValidationError):
            client.register("tiny")

    @responses.activate
    def test_register_503_when_embedder_unconfigured(self, client) -> None:
        responses.post(
            f"{BASE}/api/agents/register",
            json={"error": "embedder_unconfigured"}, status=503,
        )
        with pytest.raises(HiveMakeServerError):
            client.register("anything that meets the min length")


class TestDiscoverAgents:

    @responses.activate
    def test_discover_query_passed(self, client) -> None:
        responses.get(
            f"{BASE}/api/agents/discover",
            json={"matches": []}, status=200,
        )
        result = client.discover_agents("frontend release engineer")

        assert result == []
        assert "q=frontend" in responses.calls[0].request.url

    @responses.activate
    def test_discover_returns_matches(self, client) -> None:
        peer_id = uuid4()
        peer_project = uuid4()
        responses.get(
            f"{BASE}/api/agents/discover",
            json={"matches": [
                {
                    "agent_id": str(peer_id),
                    "project_id": str(peer_project),
                    "name": "Boudica",
                    "description": "frontend release engineer",
                    "score": 0.92,
                },
            ]}, status=200,
        )
        result = client.discover_agents("ship the build")

        assert len(result) == 1
        assert result[0].agent_id == peer_id
        assert result[0].project_id == peer_project
        assert result[0].name == "Boudica"
        assert result[0].score == 0.92

    @responses.activate
    def test_discover_limit_passed_as_query_param(self, client) -> None:
        responses.get(
            f"{BASE}/api/agents/discover",
            json={"matches": []}, status=200,
        )
        client.discover_agents("foo", limit=5)
        assert "limit=5" in responses.calls[0].request.url

    @responses.activate
    def test_discover_min_score_omitted_when_none(self, client) -> None:
        responses.get(
            f"{BASE}/api/agents/discover",
            json={"matches": []}, status=200,
        )
        client.discover_agents("foo")
        assert "min_score" not in responses.calls[0].request.url

    @responses.activate
    def test_discover_min_score_passed_when_set(self, client) -> None:
        responses.get(
            f"{BASE}/api/agents/discover",
            json={"matches": []}, status=200,
        )
        client.discover_agents("foo", min_score=0.3)
        assert "min_score=0.3" in responses.calls[0].request.url

    @responses.activate
    def test_discover_registration_required_403(self, client) -> None:
        responses.get(
            f"{BASE}/api/agents/discover",
            json={"error": "registration_required",
                  "detail": "call POST /api/agents/register first"},
            status=403,
        )
        with pytest.raises(HiveMakeForbidden):
            client.discover_agents("anything")


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------

class TestErrorMapping:

    @responses.activate
    def test_401_auth(self, client) -> None:
        responses.get(f"{BASE}/api/tickets", json={"error": "unauthorized"}, status=401)
        with pytest.raises(HiveMakeAuthError):
            client.list_inbox()

    @responses.activate
    def test_403_forbidden(self, client) -> None:
        responses.post(
            f"{BASE}/api/tickets",
            json={"error": "cross_hive_routing_not_allowed"}, status=403,
        )
        with pytest.raises(HiveMakeForbidden):
            client.file_ticket(FileTicketRequest(
                target_project_id=uuid4(),
                ticket_type=TicketType.BUG,
                title="X", description="Y",
            ))

    @responses.activate
    def test_500_server_error(self, client) -> None:
        responses.get(f"{BASE}/api/tickets", body="boom", status=500)
        with pytest.raises(HiveMakeServerError) as exc:
            client.list_inbox()
        assert exc.value.status_code == 500

    @responses.activate
    def test_unmapped_status_falls_back_to_base(self, client) -> None:
        responses.get(f"{BASE}/api/tickets", json={"error": "weird"}, status=418)
        with pytest.raises(HiveMakeAPIError) as exc:
            client.list_inbox()
        assert exc.value.status_code == 418
        # Not one of the narrow subclasses.
        assert not isinstance(exc.value, (HiveMakeAuthError, HiveMakeNotFound,
                                          HiveMakeForbidden, HiveMakeConflict,
                                          HiveMakeServerError, HiveMakeValidationError))
