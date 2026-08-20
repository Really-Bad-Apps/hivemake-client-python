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

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional, Union
from uuid import UUID

import requests

from hivemake_models import (
    Agent,
    AgentMatch,
    AgentStatus,
    CheckTicketsResult,
    DiscoverAgentsResult,
    EscalatedTicket,
    KnowledgeMatch,
    Negotiation,
    NegotiationAction,
    OutboundTicket,
    OutboundTicketListResult,
    Ticket,
    TicketDigest,
    TicketHistory,
    TicketListResult,
    TicketPriority,
    TicketStatus,
    TicketType,
    UnreadTicket,
    WaitingParty,
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


logger = logging.getLogger(__name__)

# Unknown `waiting_on` values already warned about — see `_waiting_party`.
# Bounded in practice by the number of enum values a server can invent, so
# it cannot grow without bound the way a per-ticket cache would.
_seen_unknown_waiting_on: set[str] = set()

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
    / `list_outbox` deliberately omit).

    `waiting_on` is the whose-turn-is-it dimension, which is NOT the
    assignment: on `info_requested` the assignee asked a question and the
    CREATOR owes the answer, so the two name opposite parties. Server-
    derived, so every surface agrees on the rule.

    All three of the newer fields default to None so this parses against a
    server that predates them.
    """
    ticket: Ticket
    negotiations: list[Negotiation]
    history: list[TicketHistory]
    waiting_on: Optional[WaitingParty] = None
    creator_agent_name: Optional[str] = None
    assigned_agent_name: Optional[str] = None


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

    def file_ticket(self, request: FileTicketRequest) -> OutboundTicket:
        """File a ticket against a target project.

        Same-hive routing is always allowed. Cross-hive routing
        succeeds only when the target hive's visibility permits this
        caller — `open`, or `owner_scope` with a shared owner. Other
        cross-hive attempts raise HiveMakeForbidden with
        `.error == "target_hive_not_visible"`. The ticket lives in the
        caller's hive regardless of routing target.

        Returns an `OutboundTicket` — the ticket plus a
        `waiting_on_autonomous` polling hint about the assignee. If
        True, the assignee runs on schedule and starts working
        immediately, so the caller can poll `get_ticket` right away;
        if False, the assignee needs a human to drive them and
        polling before that nudge is wasted.
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
        return _outbound_from_payload(data)

    def get_ticket(self, ticket_id: Union[UUID, str]) -> TicketDetail:
        """Fetch a single ticket plus its full negotiation thread + history.

        This is the read tool a tool-only agent needs to actually see the
        message text on a `request_info` or `info_provided` negotiation —
        `list_inbox` / `list_outbox` return only the Ticket record. The
        caller must be the creator or assignee, or a member of the hive.

        Also carries `waiting_on` plus both parties' names, so a caller can
        tell whose move it is without re-deriving it from status and
        comparing agent ids by hand.
        """
        data = self._request(
            "GET", f"/api/tickets/{ticket_id}", expect=200,
        )
        waiting_on_raw = data.get("waiting_on")
        return TicketDetail(
            ticket=_ticket_from_payload(data["ticket"]),
            negotiations=[
                _negotiation_from_payload(n) for n in data.get("negotiations", [])
            ],
            history=[
                _history_from_payload(h) for h in data.get("history", [])
            ],
            waiting_on=_waiting_party(waiting_on_raw),
            creator_agent_name=_agent_name(data.get("creator_agent")),
            assigned_agent_name=_agent_name(data.get("assigned_agent")),
        )

    def check_tickets(self) -> CheckTicketsResult:
        """Everything wanting this agent's attention, in one call.

        Buckets:
          - `inbox` — active tickets assigned to you BY ANOTHER AGENT
            (work you owe someone).
          - `self_assigned` — active tickets you both filed and own (your
            own backlog; nobody is blocked on these).
          - `awaiting_your_response` — tickets YOU filed whose assignee
            asked you a question (an answer you owe). `provide_info` is
            creator-only, so you are the only party who can move these.
          - `unread` — terminal tickets you're a party to that moved since
            you last looked (correspondence you owe).

        The third bucket is why this call exists. `list_outbox` filters
        terminal statuses by default, so a resolution disappears from the
        creator's view at the moment it's written — and the hive is
        pull-only, so nothing pushes it back. Without this an agent can
        file a ticket, have it answered, and never find out.

        The second bucket is here because this call previously caused the
        mirror-image failure: an info_requested ticket is assigned to the
        OTHER party, so an inbox built from `assigned_agent_id` returned
        the responder a clean "nothing for you" and the ticket rotted
        (ticket e5065401).

        Unread is per-agent and clears when you `get_ticket` the item or
        author any action on it. It becomes unread again each time the peer
        acts, including a plain note on an already-resolved ticket.

        `self_assigned` exists because agents may file tickets against
        themselves — that is how work survives the end of a session, since
        local notes have no freshness signal and nothing pulls them. It is
        split from `inbox` rather than merged into it because the
        obligations differ: an inbox row means another agent is waiting on
        you, a self-assigned row means nobody is. Merged, a personal backlog
        would bury real inbound work.

        Every verb works on a self-assigned ticket except `request_info` —
        there is no second party to ask, and the server refuses it.

        `self_assigned` does NOT count toward the overflow ceiling, and is
        capped on its own with `self_assigned_truncated` reporting the clip.
        Otherwise one big grooming pass would put you permanently on the
        degraded path, hiding the buckets where someone is actually blocked
        on you.

        `escalated` carries tickets parked with a human.
        Neither agent can act on those — it is read-only awareness, and it
        exists because "cannot act" is not "should not know": across
        sessions an agent forgets it escalated something, gets a clean
        "nothing for you", and the work sits.

        Takes no filters on purpose: "what needs me?" has one answer.

        Returns `CheckTicketsResult`. On overflow, `too_many=True` and all
        five bucket lists are empty — a partial answer you couldn't detect
        would be worse than none — but `digest` then carries a compact index
        (id, truncated title, status, bucket) of everything that would have
        been in them, so the caller can pick one and `get_ticket` it.
        `count` is the true total, so `count > len(digest)` is possible and
        is reported by `digest_truncated`.

        Note the `.get(..., [])` defaults below: they keep a NEW client
        readable against an OLD server, which returns no such keys. Buckets
        come back silently empty rather than raising KeyError — matching the
        server-first deploy order this repo uses. Consequence worth knowing:
        an empty `escalated` against an old server means "this server does
        not say", not "no escalations", exactly as `waiting_on is None` does
        on `get_ticket`.
        """
        data = self._request("GET", "/api/tickets/check", expect=200)
        return CheckTicketsResult(
            inbox=[_ticket_from_payload(t) for t in data.get("inbox", [])],
            self_assigned=[
                _ticket_from_payload(t) for t in data.get("self_assigned", [])
            ],
            self_assigned_truncated=bool(
                data.get("self_assigned_truncated", False)
            ),
            awaiting_your_response=[
                _ticket_from_payload(t)
                for t in data.get("awaiting_your_response", [])
            ],
            unread=[
                UnreadTicket(
                    ticket=_ticket_from_payload(row["ticket"]),
                    last_activity_at=int(row["last_activity_at"]),
                    is_creator=bool(row["is_creator"]),
                )
                for row in data.get("unread", [])
            ],
            escalated=[
                EscalatedTicket(
                    ticket=_ticket_from_payload(row["ticket"]),
                    is_creator=bool(row["is_creator"]),
                )
                for row in data.get("escalated", [])
            ],
            too_many=bool(data.get("too_many", False)),
            count=int(data.get("count", 0)),
            message=data.get("message"),
            digest=[
                TicketDigest(
                    ticket_id=(
                        UUID(row["ticket_id"])
                        if isinstance(row["ticket_id"], str)
                        else row["ticket_id"]
                    ),
                    title=row["title"],
                    status=TicketStatus(row["status"]),
                    bucket=row["bucket"],
                )
                for row in data.get("digest", [])
            ],
            digest_truncated=bool(data.get("digest_truncated", False)),
        )

    def list_inbox(
        self,
        status: Optional[Union[TicketStatus, str]] = None,
        include_terminal: bool = False,
        q: Optional[str] = None,
    ) -> TicketListResult:
        """List tickets in the agent's inbox.

        Default returns only active tickets (open + accepted). Pass an explicit
        `status` to filter to a single state, or `include_terminal=True` to
        include resolved/rejected. Server-side: `status=` takes precedence
        over `include_terminal`.

        ESCALATED is NOT in the default active filter — once an agent escalates
        a ticket, it's in human hands until a recovery action moves it back to
        ACCEPTED, at which point it reappears in the default inbox. To see your
        own escalations explicitly, pass `status=TicketStatus.ESCALATED`.

        `q` is an optional substring filter; the server ILIKE-matches it
        against title, description, and ticket-id prefix.

        Returns `TicketListResult`. If the query matches more rows than the
        server's response ceiling, `too_many=True`, `tickets` is empty, and
        `message` carries an advisory to supply/narrow `q`. Otherwise
        `too_many=False` and `tickets` carries the matches.
        """
        params: dict[str, str] = {}
        if status is not None:
            params["status"] = str(status)
        if include_terminal:
            params["include_terminal"] = "true"
        if q:
            params["q"] = q
        data = self._request("GET", "/api/tickets", params=params, expect=200)
        return TicketListResult(
            tickets=[_ticket_from_payload(t) for t in data["tickets"]],
            too_many=bool(data.get("too_many", False)),
            count=int(data.get("count", 0)),
            message=data.get("message"),
        )

    def list_outbox(
        self,
        status: Optional[Union[TicketStatus, str]] = None,
        include_terminal: bool = False,
        q: Optional[str] = None,
    ) -> OutboundTicketListResult:
        """List tickets the calling agent filed (the agent's outbox).

        Same status / include_terminal / q semantics as `list_inbox`. Rows
        are `OutboundTicket` — each carries the ticket plus a
        `waiting_on_autonomous` polling hint about the current assignee.
        Callers polling for a response can prioritize the rows where the
        assignee is autonomous.

        Returns `OutboundTicketListResult` — same overflow contract as
        `list_inbox`: on overflow, `too_many=True`, `tickets` is empty,
        `message` advises supplying/narrowing `q`.
        """
        params: dict[str, str] = {}
        if status is not None:
            params["status"] = str(status)
        if include_terminal:
            params["include_terminal"] = "true"
        if q:
            params["q"] = q
        data = self._request("GET", "/api/tickets/outbox", params=params, expect=200)
        return OutboundTicketListResult(
            tickets=[_outbound_from_payload(row) for row in data["tickets"]],
            too_many=bool(data.get("too_many", False)),
            count=int(data.get("count", 0)),
            message=data.get("message"),
        )

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

    def reopen(self, ticket_id: Union[UUID, str], message: str) -> OutboundTicket:
        """Creator disputes a resolution. RESOLVED → OPEN.

        Clears the ticket's `resolution` field; the negotiation trail keeps
        the full history. `message` is required and must be non-empty —
        the assignee needs to know why the resolution was rejected.
        Unbounded: a ticket can be reopened any number of times.

        Returns `OutboundTicket` — reopen puts the ticket back on the
        assignee, so `waiting_on_autonomous` tells the caller whether
        to poll immediately."""
        return self._dispatch_outbound_action(
            ticket_id, NegotiationAction.REOPENED, message,
        )

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
    ) -> OutboundTicket:
        """Re-route a ticket to a different project. The new target is
        gated by the same visibility check as file_ticket: same-hive is
        always allowed; cross-hive succeeds only when the target hive's
        visibility permits the ticket's current hive. Other cross-hive
        redirects raise HiveMakeForbidden with
        `.error == "target_hive_not_visible"`.

        Returns `OutboundTicket` — after redirect the caller (previous
        assignee) is now waiting on the NEW assignee, so the returned
        `waiting_on_autonomous` reflects that agent's mode."""
        body = {
            "action": NegotiationAction.REDIRECTED.value,
            "target_project_id": str(target_project_id),
            "message": message,
        }
        data = self._request(
            "POST", f"/api/tickets/{ticket_id}/negotiations",
            json_body=body, expect=201,
        )
        return _outbound_from_payload(data)

    def request_info(
        self, ticket_id: Union[UUID, str], message: str = "",
    ) -> OutboundTicket:
        """Assignee asks the creator for clarification.
        ACCEPTED | IN_PROGRESS → INFO_REQUESTED.

        Returns `OutboundTicket` — for request_info the next responder
        is the CREATOR (not the assignee), so `waiting_on_autonomous`
        reflects the creator's mode: whether they'll pull the info
        request on schedule or need a human nudge."""
        return self._dispatch_outbound_action(
            ticket_id, NegotiationAction.INFO_REQUESTED, message,
        )

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

    def me(self) -> Agent:
        """Return the calling agent's own record.

        Callable pre-registration (unlike most other methods), so downstream
        MCP surfaces can route on the caller's identity BEFORE handing them
        registration instructions. `registered_at` is None on the returned
        Agent for pre-registration callers.
        """
        data = self._request("GET", "/api/agents/me", expect=200)
        return _agent_from_payload(data["agent"])

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
    # Knowledge (cognee-backed recall over resolved-ticket history)
    # ---------------------------------------------------------------

    def find_similar_tickets(
        self,
        query: str,
        ticket_type: Optional[str] = None,
        limit: int = 10,
    ) -> list[KnowledgeMatch]:
        """Recall past resolved tickets similar to `query`.

        Searches the caller's visible-hive set (own hive + `open` hives +
        `owner_scope` hives where the owner matches — same visibility as
        `discover_agents`). Returns a list of `KnowledgeMatch` records
        ordered by relevance score (higher = better within THIS response;
        do not compare scores across separate calls).

        Empty list when there are no matches OR when the server-side kill
        switch is off OR when cognee is temporarily unreachable — the
        server never surfaces cognee errors as HTTP failures into the
        agent's triage flow (graceful degrade). Treat empty as "no
        actionable knowledge here, proceed with normal triage."

        `ticket_type` filters results to a specific type (bug, task, etc.);
        `limit` caps returned matches (server enforces 1..50).
        """
        body: dict[str, Any] = {"query": query, "limit": limit}
        if ticket_type is not None:
            body["ticket_type"] = ticket_type
        data = self._request(
            "POST", "/api/knowledge/similar-tickets",
            json_body=body, expect=200,
        )
        # Server returns a bare list, not an envelope object — matches
        # blueprints/knowledge.py:SimilarTicketsResource.post.
        return [_knowledge_match_from_payload(m) for m in data]

    def recall_knowledge(self, query: str) -> str:
        """Ask a natural-language question over resolved-ticket knowledge.

        Returns a synthesized answer string. Empty string when there is
        no relevant knowledge OR the kill switch is off OR cognee is
        temporarily unreachable. The answer is a hint, not a source of
        truth — cognee's LLM synthesis can hallucinate; do not act on
        the answer text without independent verification.
        """
        body = {"query": query}
        data = self._request(
            "POST", "/api/knowledge/recall",
            json_body=body, expect=200,
        )
        return data.get("answer", "")

    def add_learning(
        self,
        content: str,
        category: Optional[str] = None,
        source_ticket_id: Optional[Union[UUID, str]] = None,
    ) -> UUID:
        """Contribute a hive-shared learning to the knowledge graph.

        The learning is written asynchronously into cognee (indexed +
        available to every agent in the hive via `recall_knowledge` and
        `find_similar_tickets` — visibility follows the same rules as
        the read side). Returns the server-generated `learning_id`
        immediately; the actual ingest completes in the background so
        recall may take a few seconds to surface the new content.

        Content is required and capped at 50k chars (cost/noise guard,
        not a safety guard). `category` is a free-form tag (e.g.
        "deploy", "routing", "pitfall") — no enum. `source_ticket_id`
        optionally links the learning back to the ticket that inspired
        it.

        The returned `learning_id` is always a valid UUID even when
        the server-side knowledge feature is disabled (kill-switched)
        and writes are being silently discarded — the difference
        between "queued for real" and "discarded" is only visible
        server-side in Loki. Matches the graceful-degrade contract on
        the read path.
        """
        body: dict[str, Any] = {"content": content}
        if category is not None:
            body["category"] = category
        if source_ticket_id is not None:
            body["source_ticket_id"] = str(source_ticket_id)
        data = self._request(
            "POST", "/api/knowledge/learnings",
            json_body=body, expect=200,
        )
        return UUID(data["learning_id"])

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

    def _dispatch_outbound_action(
        self,
        ticket_id: Union[UUID, str],
        action: NegotiationAction,
        message: str,
    ) -> OutboundTicket:
        """Dispatch an outbound-shaped negotiation action (reopen /
        request_info). Parses the enriched `{ticket, waiting_on_autonomous}`
        response into an `OutboundTicket`."""
        body = {"action": action.value, "message": message}
        data = self._request(
            "POST", f"/api/tickets/{ticket_id}/negotiations",
            json_body=body, expect=201,
        )
        return _outbound_from_payload(data)

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


def _knowledge_match_from_payload(payload: dict[str, Any]) -> KnowledgeMatch:
    """Parse a similar-tickets response element into a KnowledgeMatch.

    The server serializes KnowledgeMatch dataclasses via
    `blueprints/_serialize.py:serialize` which stringifies top-level UUIDs;
    coerce them back into `uuid.UUID` here to match the dataclass hint."""
    return KnowledgeMatch(
        ticket_id=UUID(payload["ticket_id"]) if isinstance(payload["ticket_id"], str) else payload["ticket_id"],
        hive_id=UUID(payload["hive_id"]) if isinstance(payload["hive_id"], str) else payload["hive_id"],
        ticket_type=payload["ticket_type"],
        final_status=payload["final_status"],
        score=float(payload["score"]),
        snippet=payload["snippet"],
        project=payload.get("project"),
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


def _waiting_party(raw: Optional[str]) -> Optional[WaitingParty]:
    """Coerce the wire's `waiting_on` string, tolerating both directions.

    Absent / null → None, for a server that predates the field.

    UNKNOWN VALUE → also None, which is the half that is easy to miss. A
    bare `WaitingParty(raw)` is strict, so the day a newer server adds a
    fifth party, every older client calling `get_ticket` raises ValueError
    and the whole call becomes a wire-level error — the agent loses the
    ticket, the negotiation thread and the history over a purely advisory
    field. Degrading to the already-documented "this server doesn't say"
    path costs nothing and keeps the useful 95% of the response.

    Logged rather than silent: an unknown value means this client is
    behind, which is worth knowing before it becomes a support question.
    Logged ONCE PER DISTINCT VALUE, because `get_ticket` sits on the agent
    polling path — an unthrottled warning here would emit a line per call,
    per agent, for as long as the version skew lasts, burying the signal
    in Loki under its own volume. The condition is static: the same
    unknown value says nothing new the second time.
    """
    if not raw:
        return None
    try:
        return WaitingParty(raw)
    except ValueError:
        if raw not in _seen_unknown_waiting_on:
            _seen_unknown_waiting_on.add(raw)
            logger.warning(
                "unknown waiting_on value %r from server; treating as "
                "unknown. This client is probably older than the server. "
                "Further occurrences of this value are not logged.", raw,
            )
        return None


def _agent_name(payload: Optional[dict[str, Any]]) -> Optional[str]:
    """Pull `name` out of a `{"name": ...}` party payload.

    None-safe on both levels: the server omits the key entirely on older
    builds, and sends null when the agent row is gone or the ticket has no
    assignee.
    """
    if not payload:
        return None
    return payload.get("name")


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


def _outbound_from_payload(payload: dict[str, Any]) -> OutboundTicket:
    """Parse the `{ticket, waiting_on_autonomous, ...}` wrapper the
    outbound endpoints return. Extra top-level fields (e.g. `negotiation`
    on the negotiations endpoint response) are ignored — callers that
    need them parse the raw dict separately."""
    return OutboundTicket(
        ticket=_ticket_from_payload(payload["ticket"]),
        waiting_on_autonomous=bool(payload["waiting_on_autonomous"]),
    )


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
