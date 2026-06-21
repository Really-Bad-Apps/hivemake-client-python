"""Python client SDK for the HiveMake.ai REST API."""

from hivemake_client.client import (
    FileTicketRequest,
    HiveMakeClient,
    RegistrationResult,
    TicketDetail,
)
from hivemake_client.exceptions import (
    HiveMakeAPIError,
    HiveMakeAuthError,
    HiveMakeConfigError,
    HiveMakeConflict,
    HiveMakeError,
    HiveMakeForbidden,
    HiveMakeNotFound,
    HiveMakeServerError,
    HiveMakeValidationError,
)

__version__ = "0.1.0"

__all__ = [
    "HiveMakeClient",
    "FileTicketRequest",
    "RegistrationResult",
    "TicketDetail",
    "HiveMakeError",
    "HiveMakeConfigError",
    "HiveMakeAPIError",
    "HiveMakeAuthError",
    "HiveMakeForbidden",
    "HiveMakeNotFound",
    "HiveMakeConflict",
    "HiveMakeValidationError",
    "HiveMakeServerError",
]
