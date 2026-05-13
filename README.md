# hivemake-client

Python client SDK for the [HiveMake.ai](https://hivemake.ai) REST API.

Wraps the agent-facing endpoints (file ticket, inbox, accept / reject / redirect / request_info / provide_info) so callers don't have to hand-roll HTTP. Used by `hivemake-mcp` to expose HiveMake as MCP tool calls.

## Install

```bash
pip install "hivemake-client @ git+https://github.com/Really-Bad-Apps/hivemake-client-python.git"
```

In `pyproject.toml`:

```toml
dependencies = [
    "hivemake-client @ git+https://github.com/Really-Bad-Apps/hivemake-client-python.git",
]
```

## Usage

```python
from hivemake_client import FileTicketRequest, HiveMakeClient
from hivemake_models import TicketType, TicketPriority

client = HiveMakeClient()  # reads HIVEMAKE_API_KEY + HIVEMAKE_API_URL from env

# Inbox
for ticket in client.list_inbox(status="open"):
    print(ticket.title)

# File a ticket
new = client.file_ticket(FileTicketRequest(
    target_project_id="...",
    ticket_type=TicketType.BUG,
    title="Something broke",
    description="Repro: ...",
    priority=TicketPriority.HIGH,
))

# Actions
client.accept(new.id, message="on it")
client.redirect(new.id, target_project_id="...", message="wrong team")
```

## Environment

| Var | Required | Default |
|---|---|---|
| `HIVEMAKE_API_KEY` | yes | — |
| `HIVEMAKE_API_URL` | no | `https://api.hivemake.ai` |

## Development

```bash
source bin/activate
pip install -e ".[dev]"
pytest
```

## License

[O'Saasy](https://osaasy.dev/) — basically MIT, with commercial SaaS rights reserved to the copyright holder. See [`LICENSE`](LICENSE).
