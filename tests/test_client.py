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
        "pending_approval_from_agent_id": None,
        "pending_approval_from_user_id": None,
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
        ticket = client.reject(tid)
        assert ticket.id == tid
        assert b'"action": "rejected"' in responses.calls[0].request.body

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


class TestApprovalGates:

    @responses.activate
    def test_request_approval_sends_action_name(self, client) -> None:
        tid = uuid4()
        responses.post(
            f"{BASE}/api/tickets/{tid}/negotiations",
            json={
                "ticket": _ticket_payload(ticket_id=tid, status="pending_approval"),
                "negotiation": {"id": str(uuid4()), "action": "approval_requested"},
            },
            status=201,
        )
        ticket = client.request_approval(tid, action_name="merge_pr", message="rev?")
        assert ticket.id == tid
        body = responses.calls[0].request.body
        assert b'"action": "approval_requested"' in body
        assert b'"action_name": "merge_pr"' in body
        assert b'"message": "rev?"' in body

    @responses.activate
    def test_approve(self, client) -> None:
        tid = uuid4()
        responses.post(
            f"{BASE}/api/tickets/{tid}/negotiations",
            json={
                "ticket": _ticket_payload(ticket_id=tid, status="accepted"),
                "negotiation": {"id": str(uuid4()), "action": "approved"},
            },
            status=201,
        )
        client.approve(tid, message="lgtm")
        body = responses.calls[0].request.body
        assert b'"action": "approved"' in body
        assert b'"message": "lgtm"' in body

    @responses.activate
    def test_deny(self, client) -> None:
        tid = uuid4()
        responses.post(
            f"{BASE}/api/tickets/{tid}/negotiations",
            json={
                "ticket": _ticket_payload(ticket_id=tid, status="denied"),
                "negotiation": {"id": str(uuid4()), "action": "denied"},
            },
            status=201,
        )
        ticket = client.deny(tid)
        assert ticket.status == TicketStatus.DENIED
        assert b'"action": "denied"' in responses.calls[0].request.body

    @responses.activate
    def test_request_revision(self, client) -> None:
        tid = uuid4()
        responses.post(
            f"{BASE}/api/tickets/{tid}/negotiations",
            json={
                "ticket": _ticket_payload(ticket_id=tid, status="accepted"),
                "negotiation": {"id": str(uuid4()), "action": "revision_requested"},
            },
            status=201,
        )
        client.request_revision(tid, message="tweak")
        assert b'"action": "revision_requested"' in responses.calls[0].request.body


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
