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
    NegotiationAction,
    Ticket,
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


# UUID-typed fields on the Ticket dataclass. The server emits these as
# JSON strings; we coerce them back into UUID objects on the way in so
# `ticket.id` is a real `uuid.UUID`, matching the dataclass type hint.
_TICKET_UUID_FIELDS = (
    "id",
    "hive_id",
    "project_id",
    "created_by_agent_id",
    "assigned_agent_id",
    "pending_approval_from_agent_id",
    "pending_approval_from_user_id",
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

    def list_inbox(
        self,
        status: Optional[Union[TicketStatus, str]] = None,
    ) -> list[Ticket]:
        params: dict[str, str] = {}
        if status is not None:
            params["status"] = str(status)
        data = self._request("GET", "/api/tickets", params=params, expect=200)
        return [_ticket_from_payload(t) for t in data["tickets"]]

    # ---------------------------------------------------------------
    # Negotiation actions
    # ---------------------------------------------------------------

    def accept(self, ticket_id: Union[UUID, str], message: str = "") -> Ticket:
        return self._dispatch_action(ticket_id, NegotiationAction.ACCEPTED, message)

    def reject(self, ticket_id: Union[UUID, str], message: str = "") -> Ticket:
        return self._dispatch_action(ticket_id, NegotiationAction.REJECTED, message)

    def redirect(
        self,
        ticket_id: Union[UUID, str],
        target_project_id: Union[UUID, str],
        message: str = "",
    ) -> Ticket:
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
