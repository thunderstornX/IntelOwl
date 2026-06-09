# This file is a part of IntelOwl https://github.com/intelowlproject/IntelOwl
# See the file 'LICENSE' for copying permission.

"""LangChain callback that streams a chat turn to the WebSocket as it runs.

Attached at run level (``executor.invoke(..., config={"callbacks": [handler]})``) so it sees
both the LLM token stream and the agent's tool actions, then pushes them onto the per-user
channel group via ``group_send``. Running inside the (synchronous) Celery worker means there is
no event loop, so ``async_to_sync`` is the right bridge to the async channel layer — the same
pattern ``JobConsumer.serialize_and_send_job`` already uses.
"""

import logging

from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from langchain_core.callbacks.base import BaseCallbackHandler

from api_app.chatbot_manager.events import (
    ANSWER_MARKER,
    OutboundEvent,
    chat_group_for_user,
    status_event,
    token_event,
)

logger = logging.getLogger(__name__)


class ChatStreamingCallbackHandler(BaseCallbackHandler):
    """Streams the agent's *final answer* token-by-token, plus a status event per tool.

    A ReAct turn is several LLM calls (Thought/Action/Observation loops); only the last one
    emits the ``Final Answer:`` line. Streaming every token would leak that scaffolding, so the
    handler buffers tokens and starts forwarding only once ``ANSWER_MARKER`` appears in the
    buffer. The marker is matched as a *substring* (not a fixed token tuple) because a local
    model's tokenization of ``"Final Answer:"`` is not guaranteed. ``on_llm_start`` re-arms the
    gate for each call so a non-final iteration never streams.

    This is a best-effort live view: the worker persists ``result["output"]`` and sends it in the
    ``chat.end`` event, which the client treats as the source of truth, so a tokenization miss
    can never corrupt the stored/displayed answer.
    """

    def __init__(self, user_id: int, session_id: int):
        self._group = chat_group_for_user(user_id)
        self._session_id = session_id
        self._channel_layer = get_channel_layer()
        self._buffer = ""
        self._answer_reached = False

    def _emit(self, event: OutboundEvent) -> None:
        async_to_sync(self._channel_layer.group_send)(self._group, event.as_channel_message())

    def on_llm_start(self, *args, **kwargs) -> None:
        # Each ReAct iteration is a fresh LLM call; re-arm so only the call that produces the
        # Final Answer streams, and drop the previous iteration's buffered scaffolding.
        self._buffer = ""
        self._answer_reached = False

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self._answer_reached:
            self._emit(token_event(self._session_id, token))
            return
        self._buffer += token
        marker_index = self._buffer.find(ANSWER_MARKER)
        if marker_index == -1:
            return
        self._answer_reached = True
        # Stream whatever already trails the marker in this same chunk; lstrip so the answer
        # doesn't open with the space/newline that usually follows "Final Answer:".
        tail = self._buffer[marker_index + len(ANSWER_MARKER) :].lstrip()
        self._buffer = ""
        if tail:
            self._emit(token_event(self._session_id, tail))

    def on_agent_action(self, action, **kwargs) -> None:
        # Fired when the ReAct agent picks a tool; action.tool is the clean tool name.
        tool_name = getattr(action, "tool", "") or ""
        self._emit(status_event(self._session_id, tool_name))
