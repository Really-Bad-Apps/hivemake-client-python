"""Native async HiveMake SDK. Endpoint contracts match HiveMakeClient.

Use ``async with AsyncHiveMakeClient(...)`` or call ``await client.aclose()``.
A supplied http_client is borrowed: its owner must close it after all callers
finish. Each request supplies its own bearer; no shared auth or cookies are
used. HTTPX transport errors and cancellation propagate without retries.
"""

import os
from typing import Any, Optional, Union
from uuid import UUID

import httpx

from hivemake_models import (
    Agent,
    CheckTicketsResult,
    DiscoverAgentsResult,
    EscalatedTicket,
    KnowledgeMatch,
    NegotiationAction,
    OutboundTicket,
    OutboundTicketListResult,
    Ticket,
    TicketDigest,
    TicketListResult,
    TicketStatus,
    UnreadTicket,
    UsageReport,
    validate_not_before,
    validate_scheduled_offset,
)

from hivemake_client.client import (
    DEFAULT_BASE_URL, DEFAULT_TIMEOUT, RECALL_TIMEOUT,
    FileTicketRequest, RegistrationResult, TicketDetail, _raise_api_error,
    _agent_from_payload, _agent_match_from_payload, _agent_name, _history_from_payload, _knowledge_match_from_payload, _negotiation_from_payload, _outbound_from_payload, _ticket_from_payload, _usage_report_from_payload, _waiting_party,
)
from hivemake_client.exceptions import HiveMakeConfigError, HiveMakeNotFound


class AsyncHiveMakeClient:
    """Async counterpart of HiveMakeClient, usable within one async runtime.

    By default HTTPX permits 100 connections, keeps 20 idle connections for
    5 seconds, and waits up to ``pool_timeout`` for a free connection. These
    transport resource limits are not an admission or per-user rate policy.
    A borrowed client's owner configures its connection limits.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        *,
        http_client: Optional[httpx.AsyncClient] = None,
        pool_timeout: float = 5.0,
    ) -> None:
        resolved_key = api_key if api_key is not None else os.environ.get("HIVEMAKE_API_KEY")
        if not resolved_key:
            raise HiveMakeConfigError(
                "HIVEMAKE_API_KEY environment variable is not set, "
                "and no api_key was passed to AsyncHiveMakeClient()."
            )
        self.api_key = resolved_key
        resolved_url = base_url if base_url is not None else os.environ.get("HIVEMAKE_API_URL", DEFAULT_BASE_URL)
        self.base_url = resolved_url.rstrip("/")
        self.timeout = timeout
        self.pool_timeout = pool_timeout
        self._owns_http_client = http_client is None
        self._http_client = http_client if http_client is not None else httpx.AsyncClient()
        self._closed = False

    async def __aenter__(self) -> "AsyncHiveMakeClient":
        if self._closed:
            raise RuntimeError("AsyncHiveMakeClient is closed")
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_http_client:
            await self._http_client.aclose()
        self._closed = True

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, str]] = None,
        expect: int = 200,
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("AsyncHiveMakeClient is closed")
        budget = timeout if timeout is not None else self.timeout
        # Construct directly: borrowing an HTTP client must never inherit its
        # auth, cookies, query parameters, or other caller-specific defaults.
        request = httpx.Request(
            method, f"{self.base_url}{path}", json=json_body, params=params,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            extensions={"timeout": httpx.Timeout(budget, pool=self.pool_timeout).as_dict()},
        )
        # No automatic redirects or retries, especially for ticket writes.
        resp = await self._http_client.send(request, auth=None, follow_redirects=False)
        if resp.status_code != expect:
            try:
                body = resp.json() if resp.content else {}
            except ValueError:
                body = {}
            _raise_api_error(resp.status_code, body, resp.reason_phrase)
        return resp.json()

    async def file_ticket(self, request: FileTicketRequest) -> OutboundTicket:
        """File a ticket against a target project.

        Same-hive routing is always allowed. Cross-hive routing
        succeeds only when the target hive's visibility permits this
        caller — `open`, or `owner_scope` with a shared owner. Other
        cross-hive attempts raise HiveMakeForbidden with
        `.error == "target_hive_not_visible"`. The ticket lives in the
        caller's hive regardless of routing target.

        Returns an `OutboundTicket` — the ticket plus a
        `waiting_on_autonomous` polling hint about the assignee, plus an optional
        `suggested_poll_interval_seconds` for waiting on its first response.
        When autonomous, poll at that interval when provided; otherwise use
        backoff starting around 30 seconds. Manual agents need a human nudge,
        so polling before that nudge is wasted.
        """
        body = {
            "target_project_id": str(request.target_project_id),
            "ticket_type": str(request.ticket_type),
            "title": request.title,
            "description": request.description,
            "priority": str(request.priority),
            "message": request.message,
        }
        if request.not_before is not None:
            validate_not_before(request.not_before)
            # Older request schemas silently ignore unknown fields. Never send
            # a scheduled filing to one: it would become immediate work.
            health = await self._request("GET", "/api/health", expect=200)
            if "scheduled_tickets" not in health.get("capabilities", []):
                raise HiveMakeConfigError("Server does not support scheduled tickets; no ticket was filed")
            body["not_before"] = request.not_before
        data = await self._request("POST", "/api/tickets", json_body=body, expect=201)
        return _outbound_from_payload(data)

    async def get_ticket(self, ticket_id: Union[UUID, str]) -> TicketDetail:
        """Fetch a single ticket plus its full negotiation thread + history.

        This is the read tool a tool-only agent needs to actually see the
        message text on a `request_info` or `info_provided` negotiation —
        `list_inbox` / `list_outbox` return only the Ticket record. The
        caller must be the creator or assignee, or a member of the hive.

        Also carries `waiting_on` plus both parties' names, so a caller can
        tell whose move it is without re-deriving it from status and
        comparing agent ids by hand.
        """
        data = await self._request(
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
            is_scheduled=bool(data.get("is_scheduled", False)),
            creator_agent_name=_agent_name(data.get("creator_agent")),
            assigned_agent_name=_agent_name(data.get("assigned_agent")),
            waiting_on_last_seen_seconds=data.get("waiting_on_last_seen_seconds"),
        )

    async def check_tickets(self, scheduled_offset: int = 0) -> CheckTicketsResult:
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

        scheduled_offset pages only the creator's scheduled backlog; it
        never hides or filters the other buckets.

        Returns `CheckTicketsResult`. On overflow, `too_many=True` and all
        six bucket lists are empty — a partial answer you couldn't detect
        would be worse than none — but `digest` then carries a compact index
        (id, truncated title, status, bucket) of everything that would have
        been in them, so the caller can pick one and `get_ticket` it.
        `count` is the true total, so `count > len(digest)` is possible and
        is reported by `digest_truncated`.

        Note the `.get(..., [])` defaults below: they keep a NEW client
        readable against an OLD server, which returns no such keys. Buckets
        come back silently empty rather than raising KeyError. Consequence worth knowing:
        an empty `escalated` against an old server means "this server does
        not say", not "no escalations", exactly as `waiting_on is None` does
        on `get_ticket`.
        """
        validate_scheduled_offset(scheduled_offset)
        params = {"scheduled_offset": str(scheduled_offset)} if scheduled_offset else None
        data = await self._request("GET", "/api/tickets/check", params=params, expect=200)
        return CheckTicketsResult(
            scheduled=[_ticket_from_payload(t) for t in data.get("scheduled", [])],
            scheduled_truncated=bool(data.get("scheduled_truncated", False)),
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

    async def list_inbox(
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
        data = await self._request("GET", "/api/tickets", params=params, expect=200)
        return TicketListResult(
            tickets=[_ticket_from_payload(t) for t in data["tickets"]],
            too_many=bool(data.get("too_many", False)),
            count=int(data.get("count", 0)),
            message=data.get("message"),
        )

    async def list_outbox(
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
        data = await self._request("GET", "/api/tickets/outbox", params=params, expect=200)
        return OutboundTicketListResult(
            tickets=[_outbound_from_payload(row) for row in data["tickets"]],
            too_many=bool(data.get("too_many", False)),
            count=int(data.get("count", 0)),
            message=data.get("message"),
        )

    async def accept(self, ticket_id: Union[UUID, str], message: str = "") -> Ticket:
        return await self._dispatch_action(ticket_id, NegotiationAction.ACCEPTED, message)

    async def reject(self, ticket_id: Union[UUID, str], message: str) -> Ticket:
        """Assignee rejects the ticket. OPEN → REJECTED. Terminal.

        `message` is required and must be non-empty server-side (422).
        The creator needs a reason ("not my project", "duplicate",
        "out of scope," etc.) — empty rejections are useless to them."""
        return await self._dispatch_action(ticket_id, NegotiationAction.REJECTED, message)

    async def resolve(self, ticket_id: Union[UUID, str], message: str) -> Ticket:
        """Assignee marks the ticket as resolved. OPEN | ACCEPTED → RESOLVED.

        Soft-terminal — the creator can call reopen() to dispute. `message`
        is required and must be non-empty; it is written to the ticket's
        `resolution` field so the requester can read it without scraping
        the negotiation trail. Whitespace-only counts as empty (server
        returns 422)."""
        return await self._dispatch_action(ticket_id, NegotiationAction.RESOLVED, message)

    async def reopen(self, ticket_id: Union[UUID, str], message: str) -> OutboundTicket:
        """Creator disputes a resolution. RESOLVED → OPEN.

        Clears the ticket's `resolution` field; the negotiation trail keeps
        the full history. `message` is required and must be non-empty —
        the assignee needs to know why the resolution was rejected.
        Unbounded: a ticket can be reopened any number of times.

        Returns `OutboundTicket` — reopen puts the ticket back on the
        assignee, so `waiting_on_autonomous` tells the caller whether to poll.
        Use `suggested_poll_interval_seconds` when provided for that first
        response; it does not estimate completion time."""
        return await self._dispatch_outbound_action(
            ticket_id, NegotiationAction.REOPENED, message,
        )

    async def close(self, ticket_id: Union[UUID, str], message: str) -> Ticket:
        """Assignee marks the ticket no-fault terminal (obsolete/duplicate/won't-fix).
        OPEN | ACCEPTED → CLOSED. Distinct from reject ("not your problem")
        and resolve ("work delivered").

        `message` is required and must be non-empty server-side (422).
        The creator needs to know why no work will happen — "duplicate
        of #N", "obsolete", "scope changed," etc."""
        return await self._dispatch_action(ticket_id, NegotiationAction.CLOSED, message)

    async def reschedule(self, ticket_id: Union[UUID, str], not_before: Optional[int],
                   message: str = "") -> OutboundTicket:
        """Creator-only update while scheduled. UTC epoch seconds; null releases now."""
        validate_not_before(not_before)
        data = await self._request("POST", f"/api/tickets/{ticket_id}/negotiations",
                             json_body={"action": "rescheduled", "not_before": not_before,
                                        "message": message}, expect=201)
        return _outbound_from_payload(data)

    async def withdraw(self, ticket_id: Union[UUID, str], message: str = "") -> Ticket:
        """Creator cancels their own ticket. OPEN | ACCEPTED → WITHDRAWN.
        ESCALATED is excluded — mid-flight escalations stay with the humans
        handling them."""
        return await self._dispatch_action(ticket_id, NegotiationAction.WITHDRAWN, message)

    async def redirect(
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
        data = await self._request(
            "POST", f"/api/tickets/{ticket_id}/negotiations",
            json_body=body, expect=201,
        )
        return _outbound_from_payload(data)

    async def request_info(
        self, ticket_id: Union[UUID, str], message: str = "",
    ) -> OutboundTicket:
        """Assignee asks the creator for clarification.
        ACCEPTED | IN_PROGRESS → INFO_REQUESTED.

        Returns `OutboundTicket` — for request_info the next responder
        is the CREATOR (not the assignee), so `waiting_on_autonomous`
        reflects the creator's mode: whether they'll pull the info
        request on schedule or need a human nudge."""
        return await self._dispatch_outbound_action(
            ticket_id, NegotiationAction.INFO_REQUESTED, message,
        )

    async def cancel_info_request(self, ticket_id: Union[UUID, str], reason: str) -> Ticket:
        """Retract your pending question and resume the assigned ticket.

        Requires a non-empty reason, recorded in the thread. If a reply or
        another transition already won, raises HiveMakeConflict; read
        the ticket before continuing. Does not mark peer messages read.
        """
        return await self._dispatch_action(ticket_id, NegotiationAction.INFO_REQUEST_CANCELLED, reason)

    async def provide_info(self, ticket_id: Union[UUID, str], message: str = "") -> Ticket:
        return await self._dispatch_action(ticket_id, NegotiationAction.INFO_PROVIDED, message)

    async def add_note(self, ticket_id: Union[UUID, str], message: str) -> Ticket:
        """State-neutral note on a ticket you filed or a ticket assigned to you.

        Appends a message to the negotiation thread without any status
        transition — useful when you need to add context that doesn't fit
        an existing action (e.g. "actually change of plan, do X instead"
        after the assignee has already accepted, or "shipped a related
        fix, please retry when ready").

        Server enforces that the caller is either the current assignee OR
        the original creator. Message is required and must be non-empty.
        """
        return await self._dispatch_action(ticket_id, NegotiationAction.NOTE, message)

    async def escalate(self, ticket_id: Union[UUID, str], message: str = "") -> Ticket:
        """Escalate a stuck accepted ticket to the humans in this hive.

        Only valid when the agent is the assignee AND the ticket is in
        ACCEPTED — escalation is the "I'm mid-work and blocked" lever.
        Broadcast: every hive member sees it on the escalation queue, and
        the hive owners get a Telegram DM if linked.
        """
        return await self._dispatch_action(ticket_id, NegotiationAction.ESCALATED, message)

    async def register(self, description: str) -> RegistrationResult:
        """Register (or re-register) this agent's capabilities.

        Required before any other tool — until this call succeeds the agent
        is a "ghost" and the server returns 403 registration_required from
        every other endpoint. Idempotent: re-calling refreshes the
        description, regenerates the embedding, and re-stamps registered_at.
        """
        body = {"description": description}
        data = await self._request("POST", "/api/agents/register", json_body=body, expect=200)
        return RegistrationResult(agent=_agent_from_payload(data["agent"]))

    async def me(self) -> Agent:
        """Return the calling agent's own record.

        Callable pre-registration (unlike most other methods), so downstream
        MCP surfaces can route on the caller's identity BEFORE handing them
        registration instructions. `registered_at` is None on the returned
        Agent for pre-registration callers.
        """
        data = await self._request("GET", "/api/agents/me", expect=200)
        return _agent_from_payload(data["agent"])

    async def discover_agents(
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
        data = await self._request("GET", "/api/agents/discover", params=params, expect=200)
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

    async def find_similar_tickets(
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
        data = await self._request(
            "POST", "/api/knowledge/similar-tickets",
            json_body=body, expect=200,
        )
        # Server returns a bare list, not an envelope object — matches
        # blueprints/knowledge.py:SimilarTicketsResource.post.
        return [_knowledge_match_from_payload(m) for m in data]

    async def recall_knowledge(self, query: str) -> str:
        """Ask a natural-language question over resolved-ticket knowledge.

        Returns a synthesized answer string. Empty string when there is
        no relevant knowledge OR the kill switch is off OR cognee is
        temporarily unreachable. The answer is a hint, not a source of
        truth — cognee's LLM synthesis can hallucinate; do not act on
        the answer text without independent verification.

        Expect this call to take ~55s, and to get slower as more hives
        become visible to the caller — cognee scopes its graph and vector
        work by one tag per visible hive. Most of that time is retrieval
        and graph projection, not the answer model. It carries its own
        `RECALL_TIMEOUT` budget for that reason; see the constant for the
        full nesting of timeouts this sits inside.
        """
        body = {"query": query}
        data = await self._request(
            "POST", "/api/knowledge/recall",
            json_body=body, expect=200, timeout=RECALL_TIMEOUT,
        )
        return data.get("answer", "")

    async def admin_usage(self) -> Optional[UsageReport]:
        """Every owner's storage footprint as of the latest successful sweep.

        ADMIN ONLY — this crosses the owner boundary that every other read on
        this client respects, so the server gates it on an explicit
        allowlist. A caller that is not on it gets 403, which surfaces as
        `HiveMakeForbidden`.

        Returns None when no sweep has ever succeeded. That is distinct from
        a report with no owners: None means nothing has been measured, an
        empty `owners` list means a sweep ran and found no billable storage.
        Collapsing the two would make "the meter is broken" and "nobody is
        using anything" look identical, and those have opposite responses.

        The figures are up to a day old by design — read `measured_at` before
        quoting one at anybody.
        """
        try:
            data = await self._request("GET", "/api/admin/usage", expect=200)
        except HiveMakeNotFound as exc:
            # Discriminate on the error CODE, not the status. Two different
            # 404s reach here and they mean opposite things:
            #
            #   no_successful_run -> the meter has genuinely not produced a
            #                        figure yet. A real, reportable answer.
            #   not_found         -> the ROUTE does not exist: an older
            #                        server, a routing change, a typo.
            #
            # Collapsing them would make "this server cannot answer" look
            # exactly like "nothing has been measured" — a confident false
            # negative, which is the failure this whole surface was built to
            # avoid. Observed live on 2026-09-20: mcp :48 shipped ahead of
            # server :92 and the missing route reported as a clean
            # UsageNotMeasured.
            if exc.error_code != "no_successful_run":
                raise
            return None
        return _usage_report_from_payload(data)

    async def add_learning(
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
        data = await self._request(
            "POST", "/api/knowledge/learnings",
            json_body=body, expect=200,
        )
        return UUID(data["learning_id"])

    async def _dispatch_action(
        self,
        ticket_id: Union[UUID, str],
        action: NegotiationAction,
        message: str,
    ) -> Ticket:
        body = {"action": action.value, "message": message}
        data = await self._request(
            "POST", f"/api/tickets/{ticket_id}/negotiations",
            json_body=body, expect=201,
        )
        return _ticket_from_payload(data["ticket"])

    async def _dispatch_outbound_action(
        self,
        ticket_id: Union[UUID, str],
        action: NegotiationAction,
        message: str,
    ) -> OutboundTicket:
        """Dispatch an outbound-shaped negotiation action (reopen /
        request_info). Parses the enriched `{ticket, waiting_on_autonomous}`
        response into an `OutboundTicket`."""
        body = {"action": action.value, "message": message}
        data = await self._request(
            "POST", f"/api/tickets/{ticket_id}/negotiations",
            json_body=body, expect=201,
        )
        return _outbound_from_payload(data)
