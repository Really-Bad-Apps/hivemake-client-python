"""Python client SDK for the HiveMake.ai REST API."""

from importlib.metadata import PackageNotFoundError, version

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

# DERIVED from installed package metadata, not hand-written. This was
# hardcoded "0.1.0" through thirteen releases, so anyone asking the
# obvious question — "is this client new enough to have <field>?" — got a
# confidently wrong answer. Exactly the question the forward-compat
# handling in `_waiting_party` exists to make answerable, so a lying
# __version__ undercuts it.
#
# Deriving means it cannot drift again: bumping pyproject.toml is now the
# only place a version lives.
try:
    __version__ = version("hivemake-client")
except PackageNotFoundError:  # running from a source tree, not installed
    __version__ = "0.0.0+unknown"

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
