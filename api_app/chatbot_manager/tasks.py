# This file is a part of IntelOwl https://github.com/intelowlproject/IntelOwl
# See the file 'LICENSE' for copying permission.

import datetime
import logging

from asgiref.sync import async_to_sync
from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded
from channels.layers import get_channel_layer
from django.conf import settings
from django.contrib.auth import get_user_model
from django.db.models import Max
from django.db.models.functions import Coalesce
from django.utils.timezone import now

from api_app.chatbot_manager.events import (
    DETAIL_SESSION_NOT_FOUND,
    DETAIL_TIMEOUT,
    DETAIL_UNAVAILABLE,
    chat_group_for_user,
    end_event,
    error_event,
    start_event,
)
from intel_owl.tasks import FailureLoggedTask

logger = logging.getLogger(__name__)


# soft_time_limit is 300s on purpose: a single chat turn can chain several ReAct
# iterations, each one an Ollama inference on a local model, so it can legitimately take
# tens of seconds to a couple of minutes. A *soft* limit raises SoftTimeLimitExceeded
# (catchable, lets us clean up) rather than a hard time_limit that SIGKILLs the worker,
# and it bounds a hung Ollama call so it can't occupy a worker indefinitely.
@shared_task(base=FailureLoggedTask, soft_time_limit=300)
def process_chat_message(session_id: int, user_message: str, user_id: int) -> None:
    """Run one chat turn off-request and stream it to the user's WebSocket group.

    Enqueued by ChatConsumer.receive_json. The agent runs synchronously here (so token and
    tool callbacks can bridge to the async channel layer via async_to_sync) and pushes
    chat.start -> chat.status/chat.token* -> chat.end onto the per-user group, which the
    consumer relays to the browser. On any failure a single chat.error is emitted instead and
    the turn is dropped (no assistant message persisted), mirroring the sync REST path.

    Persistence is the source of truth: the streamed tokens are a live preview, while
    result["output"] is what gets stored and sent in chat.end. History is snapshotted before
    this turn is written so the current message is not double-counted.
    """
    # The agent/LLM stack (langchain + ChatOllama) is imported lazily so the other Celery
    # workers that load this module never pay for that heavy import — only the chatbot worker,
    # which actually runs this task, does.
    from api_app.chatbot_manager.agent.agent import build_agent_executor, format_history
    from api_app.chatbot_manager.agent.memory import DjangoChatMessageHistory
    from api_app.chatbot_manager.agent.streaming import ChatStreamingCallbackHandler
    from api_app.chatbot_manager.models import ChatMessage, ChatSession

    channel_layer = get_channel_layer()
    group = chat_group_for_user(user_id)

    def emit(event) -> None:
        async_to_sync(channel_layer.group_send)(group, event.as_channel_message())

    # Defense in depth: never trust the task args. The session must exist AND belong to the
    # user (the consumer already enforces this, but the task must not assume that).
    try:
        session = ChatSession.objects.get(pk=session_id, user_id=user_id)
    except ChatSession.DoesNotExist:
        logger.warning("process_chat_message: session %s not found for user %s", session_id, user_id)
        emit(error_event(session_id, DETAIL_SESSION_NOT_FOUND))
        return

    user = get_user_model().objects.get(pk=user_id)
    handler = ChatStreamingCallbackHandler(user_id=user_id, session_id=session_id)
    history = DjangoChatMessageHistory(session=session)
    chat_history_text = format_history(history.messages)

    emit(start_event(session_id))
    try:
        executor = build_agent_executor(user=user, streaming=True)
        result = executor.invoke(
            {"input": user_message, "chat_history": chat_history_text},
            config={"callbacks": [handler]},
        )
    except SoftTimeLimitExceeded:
        logger.warning("process_chat_message: timed out for session %s", session_id)
        emit(error_event(session_id, DETAIL_TIMEOUT))
        return
    except Exception as exc:  # noqa: BLE001 - any agent/Ollama failure must still reach the client
        logger.exception("process_chat_message: agent run failed for session %s: %s", session_id, exc)
        emit(error_event(session_id, DETAIL_UNAVAILABLE))
        return

    response_text = result.get("output", "")
    history.add_user_message(user_message)
    # Create the assistant row directly (rather than add_ai_message + a latest("timestamp")
    # re-query) so chat.end carries its exact id, without a lookup two overlapping turns of the
    # same session could race on.
    ai_message = ChatMessage.objects.create(
        session=session, role=ChatMessage.Role.ASSISTANT, content=response_text
    )
    emit(end_event(session_id, ai_message.id, response_text))


# soft_time_limit=1800: daily maintenance task whose work is a single bulk DELETE that
# cascades to ChatMessage rows. 30 minutes is generous headroom for a large backlog of stale
# sessions while still bounding a runaway query; a *soft* limit raises a catchable
# SoftTimeLimitExceeded (so the logged count stays meaningful) instead of a hard time_limit
# that SIGKILLs the worker mid-delete.
@shared_task(base=FailureLoggedTask, soft_time_limit=1800)
def delete_old_chat_sessions() -> int:
    """
    Periodic cleanup: delete ChatSessions whose last activity is older than
    CHATBOT_MESSAGE_RETENTION_DAYS. "Last activity" is the most recent message timestamp,
    falling back to created_at for sessions with no messages yet (ChatSession.updated_at is
    auto_now but is NOT touched when a ChatMessage is created, so it does not reflect activity).
    Deleting the session removes its messages via the FK on_delete=CASCADE.
    Returns the number of deleted sessions.
    """
    from api_app.chatbot_manager.models import ChatSession

    logger.info("started delete_old_chat_sessions")

    cutoff = now() - datetime.timedelta(days=settings.CHATBOT_MESSAGE_RETENTION_DAYS)
    # Coalesce(..., created_at) is required: without it, sessions with no messages get
    # last_activity = NULL and __lt never matches, so they would never be cleaned up.
    stale = ChatSession.objects.annotate(
        last_activity=Coalesce(Max("messages__timestamp"), "created_at")
    ).filter(last_activity__lt=cutoff)
    # .delete() on an annotated queryset is unreliable, so materialize the pks first and
    # delete them in a separate, un-annotated step.
    stale_pks = list(stale.values_list("pk", flat=True))
    logger.info(f"found {len(stale_pks)} stale chat sessions")
    ChatSession.objects.filter(pk__in=stale_pks).delete()  # CASCADE removes the messages

    logger.info("finished delete_old_chat_sessions")
    return len(stale_pks)
