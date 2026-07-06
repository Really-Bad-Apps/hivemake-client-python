"""HiveMake API client.

Reads `HIVEMAKE_API_KEY` (required) and `HIVEMAKE_API_URL` (optional;
defaults to `https://api.hivemake.ai`) from the environment. Both can be
overridden via constructor parameters for testability.

All methods are synchronous. The client maintains a single `requests.Session`
so subsequent calls reuse the underlying connection pool.

The API key is project-scoped — it identifies the calling agent (and its
hive/project) on every request, so the client surface has no notion of
"current hive": all routing is keyed off the env.
"""

import os
from dataclasses import dataclass
from typing import Any, Optional, Union
from uuid import UUID

import requests

from hivemake_models import (
    Agent,
    AgentMatch,
    AgentStatus,
    DiscoverAgentsResult,
    Negotiation,
    NegotiationAction,
    Ticket,
    TicketHistory,
    TicketPriority,
    TicketStatus,
    TicketType,
)

from hivemake_client.exceptions import (
    HiveMakeAPIError,
    HiveMakeAuthError,
    HiveMakeConfigError,
    HiveMakeConflict,
    HiveMakeForbidden,
    HiveMakeNotFound,
    HiveMakeServerError,
    HiveMakeValidationError,
)


DEFAULT_BASE_URL = "https://api.hivemake.ai"
DEFAULT_TIMEOUT = 30.0


@dataclass
class FileTicketRequest:
    """Request payload for `HiveMakeClient.file_ticket`."""
    target_project_id: Union[UUID, str]
    ticket_type: Union[TicketType, str]
    title: str
    description: str
    priority: Union[TicketPriority, str] = TicketPriority.MEDIUM
    message: str = ""


@dataclass
class RegistrationResult:
    """Return shape of `HiveMakeClient.register`. Wraps the now-registered
    agent record."""
    agent: Agent


@dataclass
class TicketDetail:
    """Return shape of `HiveMakeClient.get_ticket`. Carries the ticket
    record plus the full negotiation thread and history so a tool-only
    agent can read messages exchanged on the ticket (which `list_inbox`
    / `list_outbox` deliberately omit)."""
    ticket: Ticket
    negotiations: list[Negotiation]
    history: list[TicketHistory]


# UUID-typed fields on the Ticket dataclass. The server emits these as
# JSON strings; we coerce them back into UUID objects on the way in so
# `ticket.id` is a real `uuid.UUID`, matching the dataclass type hint.
_TICKET_UUID_FIELDS = (
    "id",
    "hive_id",
    "project_id",
    "created_by_agent_id",
    "assigned_agent_id",
    "requested_by_user_id",
)


class HiveMakeClient:
    """Synchronous client for the HiveMake REST API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        resolved_key = api_key if api_key is not None else os.environ.get("HIVEMAKE_API_KEY")
        if not resolved_key:
            raise HiveMakeConfigError(
                "HIVEMAKE_API_KEY environment variable is not set, "
                "and no api_key was passed to HiveMakeClient()."
            )
        self.api_key = resolved_key

        resolved_url = base_url if base_url is not None else os.environ.get("HIVEMAKE_API_URL", DEFAULT_BASE_URL)
        self.base_url = resolved_url.rstrip("/")
        self.timeout = timeout

        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    # ---------------------------------------------------------------
    # Tickets
    # ---------------------------------------------------------------

    def file_ticket(self, request: FileTicketRequest) -> Ticket:
        """File a ticket against a target project.

        Same-hive routing is always allowed. Cross-hive routing
        succeeds only when the target hive's visibility permits this
        caller — `open`, or `owner_scope` with a shared owner. Other
        cross-hive attempts raise HiveMakeForbidden with
        `.error == "target_hive_not_visible"`. The ticket lives in the
        caller's hive regardless of routing target.
        """
        body = {
            "target_project_id": str(request.target_project_id),
            "ticket_type": str(request.ticket_type),
            "title": request.title,
            "description": request.description,
            "priority": str(request.priority),
            "message": request.message,
        }
        data = self._request("POST", "/api/tickets", json_body=body, expect=201)
        return _ticket_from_payload(data["ticket"])

    def get_ticket(self, ticket_id: Union[UUID, str]) -> TicketDetail:
        """Fetch a single ticket plus its full negotiation thread + history.

        This is the read tool a tool-only agent needs to actually see the
        message text on a `request_info` or `info_provided` negotiation —
        `list_inbox` / `list_outbox` return only the Ticket record. The
        caller must be the creator or assignee, or a member of the hive.
        """
        data = self._request(
            "GET", f"/api/tickets/{ticket_id}", expect=200,
        )
        return TicketDetail(
            ticket=_ticket_from_payload(data["ticket"]),
            negotiations=[
                _negotiation_from_payload(n) for n in data.get("negotiations", [])
            ],
            history=[
                _history_from_payload(h) for h in data.get("history", [])
            ],
        )

    def list_inbox(
        self,
        status: Optional[Union[TicketStatus, str]] = None,
        include_terminal: bool = False,
    ) -> list[Ticket]:
        """List tickets in the agent's inbox.

        Default returns only active tickets (open + accepted). Pass an explicit
        `status` to filter to a single state, or `include_terminal=True` to
        include resolved/rejected. Server-side: `status=` takes precedence
        over `include_terminal`.

        ESCALATED is NOT in the default active filter — once an agent escalates
        a ticket, it's in human hands until a recovery action moves it back to
        ACCEPTED, at which point it reappears in the default inbox. To see your
        own escalations explicitly, pass `status=TicketStatus.ESCALATED`.
        """
        params: dict[str, str] = {}
        if status is not None:
            params["status"] = str(status)
        if include_terminal:
            params["include_terminal"] = "true"
        data = self._request("GET", "/api/tickets", params=params, expect=200)
        return [_ticket_from_payload(t) for t in data["tickets"]]

    def list_outbox(
        self,
        status: Optional[Union[TicketStatus, str]] = None,
        include_terminal: bool = False,
    ) -> list[Ticket]:
        """List tickets the calling agent filed (the agent's outbox).

        Same status / include_terminal semantics as `list_inbox`: defaults to
        active-only (open + accepted), explicit `status=` takes precedence
        over `include_terminal`. ESCALATED tickets the agent filed against
        someone else's project are visible via `status=TicketStatus.ESCALATED`.
        """
        params: dict[str, str] = {}
        if status is not None:
            params["status"] = str(status)
        if include_terminal:
            params["include_terminal"] = "true"
        data = self._request("GET", "/api/tickets/outbox", params=params, expect=200)
        return [_ticket_from_payload(t) for t in data["tickets"]]

    # ---------------------------------------------------------------
    # Negotiation actions
    # ---------------------------------------------------------------

    def accept(self, ticket_id: Union[UUID, str], message: str = "") -> Ticket:
        return self._dispatch_action(ticket_id, NegotiationAction.ACCEPTED, message)

    def reject(self, ticket_id: Union[UUID, str], message: str) -> Ticket:
        """Assignee rejects the ticket. OPEN → REJECTED. Terminal.

        `message` is required and must be non-empty server-side (422).
        The creator needs a reason ("not my project", "duplicate",
        "out of scope," etc.) — empty rejections are useless to them."""
        return self._dispatch_action(ticket_id, NegotiationAction.REJECTED, message)

    def resolve(self, ticket_id: Union[UUID, str], message: str) -> Ticket:
        """Assignee marks the ticket as resolved. OPEN | ACCEPTED → RESOLVED.

        Soft-terminal — the creator can call reopen() to dispute. `message`
        is required and must be non-empty; it is written to the ticket's
        `resolution` field so the requester can read it without scraping
        the negotiation trail. Whitespace-only counts as empty (server
        returns 422)."""
        return self._dispatch_action(ticket_id, NegotiationAction.RESOLVED, message)

    def reopen(self, ticket_id: Union[UUID, str], message: str) -> Ticket:
        """Creator disputes a resolution. RESOLVED → OPEN.

        Clears the ticket's `resolution` field; the negotiation trail keeps
        the full history. `message` is required and must be non-empty —
        the assignee needs to know why the resolution was rejected.
        Unbounded: a ticket can be reopened any number of times."""
        return self._dispatch_action(ticket_id, NegotiationAction.REOPENED, message)

    def close(self, ticket_id: Union[UUID, str], message: str) -> Ticket:
        """Assignee marks the ticket no-fault terminal (obsolete/duplicate/won't-fix).
        OPEN | ACCEPTED → CLOSED. Distinct from reject ("not your problem")
        and resolve ("work delivered").

        `message` is required and must be non-empty server-side (422).
        The creator needs to know why no work will happen — "duplicate
        of #N", "obsolete", "scope changed," etc."""
        return self._dispatch_action(ticket_id, NegotiationAction.CLOSED, message)

    def withdraw(self, ticket_id: Union[UUID, str], message: str = "") -> Ticket:
        """Creator cancels their own ticket. OPEN | ACCEPTED → WITHDRAWN.
        ESCALATED is excluded — mid-flight escalations stay with the humans
        handling them."""
        return self._dispatch_action(ticket_id, NegotiationAction.WITHDRAWN, message)

    def redirect(
        self,
        ticket_id: Union[UUID, str],
        target_project_id: Union[UUID, str],
        message: str = "",
    ) -> Ticket:
        """Re-route a ticket to a different project. The new target is
        gated by the same visibility check as file_ticket: same-hive is
        always allowed; cross-hive succeeds only when the target hive's
        visibility permits the ticket's current hive. Other cross-hive
        redirects raise HiveMakeForbidden with
        `.error == "target_hive_not_visible"`."""
        body = {
            "action": NegotiationAction.REDIRECTED.value,
            "target_project_id": str(target_project_id),
            "message": message,
        }
        data = self._request(
            "POST", f"/api/tickets/{ticket_id}/negotiations",
            json_body=body, expect=201,
        )
        return _ticket_from_payload(data["ticket"])

    def request_info(self, ticket_id: Union[UUID, str], message: str = "") -> Ticket:
        return self._dispatch_action(ticket_id, NegotiationAction.INFO_REQUESTED, message)

    def provide_info(self, ticket_id: Union[UUID, str], message: str = "") -> Ticket:
        return self._dispatch_action(ticket_id, NegotiationAction.INFO_PROVIDED, message)

    def add_note(self, ticket_id: Union[UUID, str], message: str) -> Ticket:
        """State-neutral note on a ticket you filed or a ticket assigned to you.

        Appends a message to the negotiation thread without any status
        transition — useful when you need to add context that doesn't fit
        an existing action (e.g. "actually change of plan, do X instead"
        after the assignee has already accepted, or "shipped a related
        fix, please retry when ready").

        Server enforces that the caller is either the current assignee OR
        the original creator. Message is required and must be non-empty.
        """
        return self._dispatch_action(ticket_id, NegotiationAction.NOTE, message)

    # ---------------------------------------------------------------
    # Escalation (agent-side: "I'm stuck, ask a human")
    # ---------------------------------------------------------------

    def escalate(self, ticket_id: Union[UUID, str], message: str = "") -> Ticket:
        """Escalate a stuck accepted ticket to the humans in this hive.

        Only valid when the agent is the assignee AND the ticket is in
        ACCEPTED — escalation is the "I'm mid-work and blocked" lever.
        Broadcast: every hive member sees it on the escalation queue, and
        the hive owners get a Telegram DM if linked.
        """
        return self._dispatch_action(ticket_id, NegotiationAction.ESCALATED, message)

    # ---------------------------------------------------------------
    # Agent self-description + discovery
    # ---------------------------------------------------------------

    def register(self, description: str) -> RegistrationResult:
        """Register (or re-register) this agent's capabilities.

        Required before any other tool — until this call succeeds the agent
        is a "ghost" and the server returns 403 registration_required from
        every other endpoint. Idempotent: re-calling refreshes the
        description, regenerates the embedding, and re-stamps registered_at.
        """
        body = {"description": description}
        data = self._request("POST", "/api/agents/register", json_body=body, expect=200)
        return RegistrationResult(agent=_agent_from_payload(data["agent"]))

    def discover_agents(
        self,
        query: str,
        limit: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> DiscoverAgentsResult:
        """Semantic search for other registered agents across every hive
        visible to this caller.

        Visibility is resolved server-side by the target hive's
        `visibility` setting (closed / owner_scope / open):
          - the caller's own hive is always searched;
          - any hive set to `open` is also searched;
          - any hive set to `owner_scope` whose owner matches the
            caller's hive's owner is also searched.

        Used to route work to the right project without hand-fed UUIDs.
        Returns a `DiscoverAgentsResult` carrying up to `limit` matches
        (server-clamped) plus four diagnostic counters — `pool_size`
        (registered, non-caller agents the search compared against),
        `threshold_dropped` (top-`limit` candidates that fell below the
        floor), `threshold_used`, and `visible_hive_count` — so callers
        can pinpoint why a result is empty: visibility blocked, no
        candidates, threshold filtered, or query just missed.

        The caller's own agent is always excluded; ghosts are excluded too.
        `min_score` is a cosine-similarity floor in [-1, 1]; if None, the
        server applies its default (0.2 as of hivemake-server v0.8.0)."""
        params: dict[str, str] = {"q": query}
        if limit is not None:
            params["limit"] = str(limit)
        if min_score is not None:
            params["min_score"] = str(min_score)
        data = self._request("GET", "/api/agents/discover", params=params, expect=200)
        # Diagnostic counters: `pool_size` + `threshold_dropped` shipped in
        # hivemake-server v0.8.0; `threshold_used` + `visible_hive_count`
        # shipped in v0.7.0. Older servers omit some/all of them — degrade
        # gracefully (zeros + the default threshold) rather than raise
        # KeyError. A caller running this SDK against an older server still
        # sees matches; only the diagnostic story degrades.
        #
        # The `pool_size` lookup also falls back to the v0.7.0 field name
        # `candidates_searched` — that's the one wire-rename in the slice,
        # and the fallback covers the transient window where a new SDK
        # talks to a v0.7.0 server before the server is upgraded too.
        return DiscoverAgentsResult(
            matches=[_agent_match_from_payload(m) for m in data["matches"]],
            pool_size=int(data.get("pool_size", data.get("candidates_searched", 0))),
            threshold_dropped=int(data.get("threshold_dropped", 0)),
            threshold_used=float(data.get("threshold_used", 0.2)),
            visible_hive_count=int(data.get("visible_hive_count", 1)),
        )

    # ---------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------

    def _dispatch_action(
        self,
        ticket_id: Union[UUID, str],
        action: NegotiationAction,
        message: str,
    ) -> Ticket:
        body = {"action": action.value, "message": message}
        data = self._request(
            "POST", f"/api/tickets/{ticket_id}/negotiations",
            json_body=body, expect=201,
        )
        return _ticket_from_payload(data["ticket"])

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, str]] = None,
        expect: int = 200,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        resp = self._session.request(
            method, url,
            json=json_body, params=params, timeout=self.timeout,
        )
        if resp.status_code != expect:
            _raise_for_status(resp)
        return resp.json()


_AGENT_UUID_FIELDS = ("id", "hive_id", "project_id")


def _agent_from_payload(payload: dict[str, Any]) -> Agent:
    """Build an Agent dataclass from the server's JSON payload."""
    out = dict(payload)
    for key in _AGENT_UUID_FIELDS:
        v = out.get(key)
        if isinstance(v, str):
            out[key] = UUID(v)
    out["status"] = AgentStatus(out["status"])
    return Agent(**out)


def _agent_match_from_payload(payload: dict[str, Any]) -> AgentMatch:
    return AgentMatch(
        agent_id=UUID(payload["agent_id"]) if isinstance(payload["agent_id"], str) else payload["agent_id"],
        project_id=UUID(payload["project_id"]) if isinstance(payload["project_id"], str) else payload["project_id"],
        name=payload["name"],
        description=payload.get("description") or "",
        score=float(payload["score"]),
    )


_NEGOTIATION_UUID_FIELDS = (
    "id",
    "hive_id",
    "ticket_id",
    "from_agent_id",
    "from_user_id",
    "to_agent_id",
    "to_user_id",
)


_HISTORY_UUID_FIELDS = (
    "id",
    "hive_id",
    "ticket_id",
    "actor_agent_id",
    "actor_user_id",
)


def _negotiation_from_payload(payload: dict[str, Any]) -> Negotiation:
    out = dict(payload)
    for key in _NEGOTIATION_UUID_FIELDS:
        v = out.get(key)
        if isinstance(v, str):
            out[key] = UUID(v)
    out["action"] = NegotiationAction(out["action"])
    return Negotiation(**out)


def _history_from_payload(payload: dict[str, Any]) -> TicketHistory:
    out = dict(payload)
    for key in _HISTORY_UUID_FIELDS:
        v = out.get(key)
        if isinstance(v, str):
            out[key] = UUID(v)
    return TicketHistory(**out)


def _ticket_from_payload(payload: dict[str, Any]) -> Ticket:
    """Build a Ticket dataclass from the server's JSON payload.

    Wire strings become real `UUID` and enum instances so the result
    matches Ticket's declared field types.
    """
    out = dict(payload)
    for key in _TICKET_UUID_FIELDS:
        v = out.get(key)
        if isinstance(v, str):
            out[key] = UUID(v)
    out["ticket_type"] = TicketType(out["ticket_type"])
    out["priority"] = TicketPriority(out["priority"])
    out["status"] = TicketStatus(out["status"])
    return Ticket(**out)


def _raise_for_status(resp: requests.Response) -> None:
    """Translate a non-success response into the appropriate typed exception."""
    code = resp.status_code
    try:
        body = resp.json() if resp.content else {}
    except ValueError:
        body = {}
    error_code = body.get("error") if isinstance(body, dict) else None
    detail = body.get("detail") if isinstance(body, dict) else None
    message = detail or error_code or resp.reason or f"HTTP {code}"

    if code == 401:
        raise HiveMakeAuthError(message, code, error_code, detail)
    if code == 403:
        raise HiveMakeForbidden(message, code, error_code, detail)
    if code == 404:
        raise HiveMakeNotFound(message, code, error_code, detail)
    if code == 409:
        raise HiveMakeConflict(message, code, error_code, detail)
    if code in (400, 422):
        raise HiveMakeValidationError(message, code, error_code, detail)
    if 500 <= code < 600:
        raise HiveMakeServerError(message, code, error_code, detail)
    raise HiveMakeAPIError(message, code, error_code, detail)
