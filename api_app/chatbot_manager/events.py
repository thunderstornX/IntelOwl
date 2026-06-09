# This file is a part of IntelOwl https://github.com/intelowlproject/IntelOwl
# See the file 'LICENSE' for copying permission.

"""Wire protocol for the chat WebSocket.

Single source of truth for everything that crosses the chat WebSocket so the producer
(Celery worker / streaming callback) and the consumer never hand-build event dicts:

- the Channels ``group_send`` routing types (``chat.token`` -> ``ChatConsumer.chat_token``);
- the client-facing payload built by the typed ``*_payload`` helpers;
- the ``OutboundEvent`` dataclass that pairs a routing type with its payload;
- the per-user group name and the inbound limits.

A chat turn streams to a per-user group (``chat-<user_id>``) rather than a per-session group:
the fixed URL ``ws/chat/?context_url=...`` carries no session id, the group key is the
server-authenticated user id (so no cross-tenant subscription is possible), and every payload
carries ``session_id`` so a client with several open sessions/tabs can demultiplex.
"""

from dataclasses import dataclass
from typing import Optional

# --- Channels group_send "type" -> ChatConsumer handler (dots become underscores) ---
EVENT_START = "chat.start"
EVENT_STATUS = "chat.status"
EVENT_TOKEN = "chat.token"
EVENT_END = "chat.end"
EVENT_ERROR = "chat.error"

# --- client-facing payload "type" labels (what the browser switches on) ---
CLIENT_ACK = "ack"
CLIENT_START = "start"
CLIENT_STATUS = "status"
CLIENT_TOKEN = "token"
CLIENT_END = "end"
CLIENT_ERROR = "error"

CHAT_GROUP_PREFIX = "chat-"

# The ReAct prompt ends every turn with a "Final Answer:" line (see REACT_PROMPT). The
# streaming callback streams only what follows this marker, hiding the Thought/Action scaffolding.
ANSWER_MARKER = "Final Answer:"

# Inbound guardrail: mirrors MessageRequestSerializer(message=CharField(max_length=4096)).
MAX_INBOUND_MESSAGE_LEN = 4096

# User-facing error details (kept generic on purpose: never leak internals to the client).
DETAIL_SESSION_NOT_FOUND = "Chat session not found."
DETAIL_INVALID_MESSAGE = "Invalid message payload."
DETAIL_TIMEOUT = "The assistant took too long to respond. Please try again."
DETAIL_UNAVAILABLE = "The assistant is currently unavailable. Please try again."


def chat_group_for_user(user_id: int) -> str:
    """Return the per-user channel group that carries this user's chat stream."""
    return f"{CHAT_GROUP_PREFIX}{user_id}"


def _payload(client_type: str, **fields) -> dict:
    return {"type": client_type, **fields}


# --- client-facing payload builders (also used for messages the consumer sends directly) ---
def ack_payload(session_id: int) -> dict:
    """Synchronous reply to an inbound frame, telling the client which session it landed on."""
    return _payload(CLIENT_ACK, session_id=session_id)


def start_payload(session_id: int) -> dict:
    return _payload(CLIENT_START, session_id=session_id)


def status_payload(session_id: int, tool: str) -> dict:
    return _payload(CLIENT_STATUS, session_id=session_id, tool=tool)


def token_payload(session_id: int, content: str) -> dict:
    return _payload(CLIENT_TOKEN, session_id=session_id, content=content)


def end_payload(session_id: int, message_id: int, content: str) -> dict:
    return _payload(CLIENT_END, session_id=session_id, message_id=message_id, content=content)


def error_payload(session_id: Optional[int], detail: str) -> dict:
    return _payload(CLIENT_ERROR, session_id=session_id, detail=detail)


@dataclass(frozen=True)
class OutboundEvent:
    """A chat event pushed worker -> channel layer -> consumer -> client.

    ``channel_type`` routes the ``group_send`` to the matching ``ChatConsumer`` handler;
    ``payload`` is what that handler forwards verbatim to the browser via ``send_json``.
    """

    channel_type: str
    payload: dict

    def as_channel_message(self) -> dict:
        return {"type": self.channel_type, "payload": self.payload}


# --- OutboundEvent builders (group-routed events emitted by the worker / callback) ---
def start_event(session_id: int) -> OutboundEvent:
    return OutboundEvent(EVENT_START, start_payload(session_id))


def status_event(session_id: int, tool: str) -> OutboundEvent:
    return OutboundEvent(EVENT_STATUS, status_payload(session_id, tool))


def token_event(session_id: int, content: str) -> OutboundEvent:
    return OutboundEvent(EVENT_TOKEN, token_payload(session_id, content))


def end_event(session_id: int, message_id: int, content: str) -> OutboundEvent:
    return OutboundEvent(EVENT_END, end_payload(session_id, message_id, content))


def error_event(session_id: Optional[int], detail: str) -> OutboundEvent:
    return OutboundEvent(EVENT_ERROR, error_payload(session_id, detail))
