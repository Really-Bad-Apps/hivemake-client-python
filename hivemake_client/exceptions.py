"""Exception hierarchy for hivemake-client.

Callers can catch the broad `HiveMakeError` to handle any client failure,
or narrow on specific subclasses to react to particular failure modes.
"""

from typing import Optional


class HiveMakeError(Exception):
    """Base class for every exception raised by this client."""


class HiveMakeConfigError(HiveMakeError):
    """Raised at client construction when required config is missing."""


class HiveMakeAPIError(HiveMakeError):
    """Base class for any non-2xx response from the HiveMake server.

    The `status_code` is always set; `error_code` and `detail` may be None if
    the server returned a non-JSON body (e.g. an upstream 502).
    """

    def __init__(
        self,
        message: str,
        status_code: int,
        error_code: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.detail = detail


class HiveMakeAuthError(HiveMakeAPIError):
    """401 — bad / revoked / expired API key."""


class HiveMakeForbidden(HiveMakeAPIError):
    """403 — e.g. cross-hive routing, unauthorized actor for the action."""


class HiveMakeNotFound(HiveMakeAPIError):
    """404 — e.g. ticket / target project / target agent not found."""


class HiveMakeConflict(HiveMakeAPIError):
    """409 — e.g. invalid state-machine transition."""


class HiveMakeValidationError(HiveMakeAPIError):
    """400 / 422 — request validation failed."""


class HiveMakeServerError(HiveMakeAPIError):
    """5xx — server-side failure."""
