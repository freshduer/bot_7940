"""Message bus wrapper that appends user/assistant turns to Redis lists."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.session.manager import SessionManager

_UNIFIED_SK = "unified:default"


class RedisLoggingMessageBus(MessageBus):
    """Extends MessageBus: logs inbound user text and outbound assistant replies to Redis."""

    def __init__(
        self,
        *,
        redis_client: Any,
        key_prefix: str,
        max_text_chars: int = 16_384,
        session_manager: SessionManager | None = None,
        unified_session: bool = False,
    ) -> None:
        super().__init__()
        self._redis = redis_client
        self._key_prefix = key_prefix.rstrip(":")
        self._max_text_chars = max_text_chars
        self._session_manager = session_manager
        self._unified_session = unified_session
        self._stream_buffers: dict[str, str] = {}

    def _session_key_inbound(self, msg: InboundMessage) -> str:
        if self._unified_session and not msg.session_key_override:
            return _UNIFIED_SK
        return msg.session_key

    def _session_key_outbound(self, msg: OutboundMessage) -> str:
        m = msg.metadata or {}
        if self._unified_session and m.get("message_thread_id") is None:
            return _UNIFIED_SK
        if m.get("message_thread_id") is not None and msg.channel == "telegram":
            tid = m["message_thread_id"]
            return f"{msg.channel}:{msg.chat_id}:topic:{tid}"
        return f"{msg.channel}:{msg.chat_id}"

    def _epoch(self, session_key: str) -> int:
        if self._session_manager is None:
            return 0
        try:
            s = self._session_manager.get_or_create(session_key)
            return int(s.metadata.get("conversation_log_epoch", 0) or 0)
        except Exception:
            return 0

    def _redis_key(self, session_key: str) -> str:
        epoch = self._epoch(session_key)
        safe = session_key.replace(":", "_")
        return f"{self._key_prefix}:{safe}:{epoch}"

    def _clip(self, text: str) -> str:
        if len(text) <= self._max_text_chars:
            return text
        return text[: self._max_text_chars] + "…"

    async def _rpush_json(self, session_key: str, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        try:
            await self._redis.rpush(self._redis_key(session_key), line)
        except Exception:
            logger.exception("Redis conversation log write failed")

    async def publish_inbound(self, msg: InboundMessage) -> None:
        await super().publish_inbound(msg)
        ts = msg.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        sk = self._session_key_inbound(msg)
        await self._rpush_json(
            sk,
            {
                "ts": ts.astimezone(timezone.utc).isoformat(),
                "role": "user",
                "sender_id": msg.sender_id,
                "text": self._clip(msg.content),
            },
        )

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        await super().publish_outbound(msg)
        sk = self._session_key_outbound(msg)
        meta = msg.metadata or {}
        stream_id = meta.get("_stream_id")
        if meta.get("_stream_delta") and stream_id is not None:
            self._stream_buffers[stream_id] = self._stream_buffers.get(stream_id, "") + (
                msg.content or ""
            )
            return
        if meta.get("_stream_end") and stream_id is not None:
            text = self._stream_buffers.pop(stream_id, "")
            if text:
                await self._rpush_json(
                    sk,
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "role": "assistant",
                        "text": self._clip(text),
                    },
                )
            return
        if meta.get("_tool_hint") and meta.get("_progress"):
            return
        if meta.get("_stream_delta"):
            return
        # Final OutboundMessage after streaming already logged assistant text at _stream_end.
        if meta.get("_streamed"):
            return
        text = msg.content or ""
        if not text.strip():
            return
        await self._rpush_json(
            sk,
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "role": "assistant",
                "text": self._clip(text),
            },
        )

    async def aclose(self) -> None:
        close = getattr(self._redis, "aclose", None)
        if callable(close):
            await close()
            return
        close_sync = getattr(self._redis, "close", None)
        if callable(close_sync):
            close_sync()


def try_create_redis_logging_bus(
    config: Any,
    *,
    session_manager: SessionManager | None = None,
    unified_session: bool = False,
) -> RedisLoggingMessageBus | None:
    """Return a Redis-backed bus if conversation logging is enabled and host is set."""
    from nanobot.config.schema import Config

    if not isinstance(config, Config):
        return None
    r = config.conversation_log.redis
    if not r.enabled or not (r.host or "").strip():
        return None
    password = (r.password or "").strip() or os.environ.get("REDIS_PASSWORD", "")
    if not password:
        logger.warning("conversation_log.redis enabled but password is empty; use config or REDIS_PASSWORD")
        return None

    try:
        import redis.asyncio as redis_ai
    except ImportError:
        logger.warning("redis package not installed; pip install redis")
        return None

    conn_kw: dict[str, Any] = {
        "host": r.host.strip(),
        "port": r.port,
        "password": password,
        "ssl": r.ssl,
        "decode_responses": True,
    }
    user = (r.username or "").strip()
    if user:
        conn_kw["username"] = user
    client = redis_ai.Redis(**conn_kw)
    return RedisLoggingMessageBus(
        redis_client=client,
        key_prefix=r.key_prefix,
        max_text_chars=r.max_text_chars,
        session_manager=session_manager,
        unified_session=unified_session,
    )


def make_message_bus(
    config: Any,
    *,
    session_manager: Any | None = None,
    unified_session: bool | None = None,
) -> MessageBus:
    """Return Redis logging bus when configured, otherwise a plain MessageBus."""
    u = (
        unified_session
        if unified_session is not None
        else config.agents.defaults.unified_session
    )
    bus = try_create_redis_logging_bus(config, session_manager=session_manager, unified_session=u)
    if bus is not None:
        return bus
    return MessageBus()
