"""
Database Persistence Layer for AI Voice Calling Assistant.
Manages asynchronous SQLite storage in WAL mode with aiosqlite,
providing relational persistence for calls, transcripts, messages, and summaries.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import aiosqlite

logger = logging.getLogger("server.database")

SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;

CREATE TABLE IF NOT EXISTS calls (
    id TEXT PRIMARY KEY,
    direction TEXT NOT NULL DEFAULT 'inbound' CHECK(direction IN ('inbound', 'outbound', 'simulation')),
    caller_id TEXT DEFAULT '',
    caller_name TEXT DEFAULT '',
    callee_id TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'ringing' CHECK(status IN ('initiated', 'ringing', 'active', 'completed', 'declined', 'failed', 'transferred', 'on_hold_transfer', 'on_hold', 'dialing', 'operator_talking', 'idle')),
    approval_action TEXT DEFAULT 'pending' CHECK(approval_action IN ('pending', 'ai_takeover', 'answered_personally', 'declined', 'auto_approved', 'na')),
    driver_type TEXT DEFAULT 'virtual',
    goal_prompt TEXT DEFAULT '',
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    answered_at TIMESTAMP,
    ended_at TIMESTAMP,
    duration REAL DEFAULT 0.0,
    duration_seconds INTEGER DEFAULT 0,
    hangup_reason TEXT DEFAULT '',
    end_reason TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS transcripts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id TEXT NOT NULL REFERENCES calls(id) ON DELETE CASCADE,
    speaker TEXT NOT NULL CHECK(speaker IN ('caller', 'gemini', 'agent', 'operator', 'system')),
    turn_id TEXT DEFAULT '',
    text TEXT NOT NULL,
    is_final INTEGER DEFAULT 1,
    sequence_number INTEGER DEFAULT 0,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id TEXT REFERENCES calls(id) ON DELETE CASCADE,
    caller_name TEXT NOT NULL,
    phone_number TEXT DEFAULT '',
    message TEXT NOT NULL,
    urgency TEXT NOT NULL DEFAULT 'normal' CHECK(urgency IN ('low', 'normal', 'medium', 'high', 'critical')),
    is_read INTEGER DEFAULT 0,
    action_taken TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id TEXT NOT NULL UNIQUE REFERENCES calls(id) ON DELETE CASCADE,
    summary TEXT NOT NULL,
    action_items TEXT DEFAULT '[]',
    sentiment TEXT DEFAULT 'neutral',
    key_topics TEXT DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_calls_started_at ON calls(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_calls_status ON calls(status);
CREATE INDEX IF NOT EXISTS idx_transcripts_call_id ON transcripts(call_id);
CREATE INDEX IF NOT EXISTS idx_messages_call_id ON messages(call_id);
CREATE INDEX IF NOT EXISTS idx_messages_urgency ON messages(urgency);
CREATE INDEX IF NOT EXISTS idx_messages_is_read ON messages(is_read);
CREATE INDEX IF NOT EXISTS idx_summaries_call_id ON summaries(call_id);
"""


class DatabaseManager:
    """
    Asynchronous SQLite Database Manager using aiosqlite with WAL mode,
    relational foreign key enforcement, cascading deletes, and JSON serialization.
    """

    def __init__(self, db_path: str = "voice_caller.db"):
        self.db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()
        self._initialized = False

    async def _apply_pragmas(self, conn: aiosqlite.Connection) -> None:
        """Applies essential connection-level PRAGMAs."""
        await conn.execute("PRAGMA journal_mode = WAL;")
        await conn.execute("PRAGMA synchronous = NORMAL;")
        await conn.execute("PRAGMA foreign_keys = ON;")
        await conn.execute("PRAGMA busy_timeout = 5000;")

    async def _get_conn(self) -> aiosqlite.Connection:
        """Retrieves or creates the persistent aiosqlite connection."""
        if self._conn is None:
            parent = os.path.dirname(self.db_path)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)
            self._conn = await aiosqlite.connect(self.db_path)
            self._conn.row_factory = aiosqlite.Row
            await self._apply_pragmas(self._conn)

        if not self._initialized:
            await self._conn.executescript(SCHEMA_SQL)
            await self._conn.commit()
            self._initialized = True

        return self._conn

    async def init_db(self) -> None:
        """Initializes tables, indexes, and PRAGMA settings."""
        async with self._lock:
            conn = await self._get_conn()
            await self._apply_pragmas(conn)
            await conn.executescript(SCHEMA_SQL)
            await conn.commit()
            self._initialized = True
            logger.info(f"Database initialized in WAL mode at: {self.db_path}")

    async def close(self) -> None:
        """Closes the underlying aiosqlite connection cleanly."""
        async with self._lock:
            if self._conn is not None:
                try:
                    await self._conn.close()
                except Exception as e:
                    logger.warning(f"Error closing database connection: {e}")
                finally:
                    self._conn = None
                    self._initialized = False
                    logger.info("Database connection closed.")

    async def __aenter__(self) -> "DatabaseManager":
        await self.init_db()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def create_call(self, call_data: Dict[str, Any]) -> str:
        """
        Creates a new call record in the database.
        Returns the generated or supplied call_id.
        """
        call_id = call_data.get("id") or f"call_{uuid.uuid4().hex[:12]}"
        direction = str(call_data.get("direction", "inbound")).lower()
        if direction not in ("inbound", "outbound", "simulation"):
            direction = "inbound"

        caller_id = str(call_data.get("caller_id") or "")
        caller_name = str(call_data.get("caller_name") or "")
        callee_id = str(call_data.get("callee_id") or "")

        status = str(call_data.get("status", "ringing")).lower()
        valid_statuses = {"initiated", "ringing", "active", "completed", "declined", "failed", "transferred", "on_hold_transfer", "on_hold", "dialing", "operator_talking", "idle"}
        if status not in valid_statuses:
            status = "ringing"

        approval_action = str(call_data.get("approval_action", "pending")).lower()
        valid_approvals = {"pending", "ai_takeover", "answered_personally", "declined", "auto_approved", "na"}
        if approval_action not in valid_approvals:
            approval_action = "pending"

        driver_type = str(call_data.get("driver_type", "virtual"))
        goal_prompt = str(call_data.get("goal_prompt") or "")
        started_at = call_data.get("started_at") or datetime.now(timezone.utc).isoformat()

        async with self._lock:
            conn = await self._get_conn()
            try:
                await conn.execute(
                    """
                    INSERT INTO calls (
                        id, direction, caller_id, caller_name, callee_id, status,
                        approval_action, driver_type, goal_prompt, started_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (call_id, direction, caller_id, caller_name, callee_id, status, approval_action, driver_type, goal_prompt, started_at)
                )
                await conn.commit()
                return call_id
            except Exception:
                await conn.rollback()
                raise

    async def update_call_status(self, call_id: str, status: str, **kwargs) -> bool:
        """
        Updates the status and optional lifecycle metrics of an existing call.
        Synchronizes hangup_reason/end_reason and duration/duration_seconds.
        """
        status = str(status).lower()
        hangup_reason = kwargs.get("hangup_reason") or kwargs.get("end_reason") or ""
        duration_val = kwargs.get("duration")
        if duration_val is None:
            duration_val = kwargs.get("duration_seconds", 0.0)
        try:
            duration_float = float(duration_val)
        except (ValueError, TypeError):
            duration_float = 0.0
        duration_int = int(round(duration_float))

        approval_action = kwargs.get("approval_action")
        caller_name = kwargs.get("caller_name")

        async with self._lock:
            conn = await self._get_conn()
            try:
                updates = ["status = ?"]
                params: List[Any] = [status]

                if status in ("completed", "failed", "declined"):
                    updates.append("ended_at = CURRENT_TIMESTAMP")
                elif kwargs.get("ended_at"):
                    updates.append("ended_at = ?")
                    params.append(kwargs["ended_at"])

                if kwargs.get("answered_at"):
                    updates.append("answered_at = ?")
                    params.append(kwargs["answered_at"])

                if "hangup_reason" in kwargs or "end_reason" in kwargs:
                    updates.append("hangup_reason = ?")
                    params.append(hangup_reason)
                    updates.append("end_reason = ?")
                    params.append(hangup_reason)

                if "duration" in kwargs or "duration_seconds" in kwargs:
                    updates.append("duration = ?")
                    params.append(duration_float)
                    updates.append("duration_seconds = ?")
                    params.append(duration_int)

                if approval_action is not None:
                    updates.append("approval_action = ?")
                    params.append(approval_action)

                if caller_name is not None:
                    updates.append("caller_name = ?")
                    params.append(caller_name)

                params.append(call_id)
                query = f"UPDATE calls SET {', '.join(updates)} WHERE id = ?"
                cursor = await conn.execute(query, tuple(params))
                await conn.commit()
                return cursor.rowcount > 0
            except Exception:
                await conn.rollback()
                raise

    async def append_transcript(
        self,
        call_id: str,
        speaker: str,
        text: str,
        is_final: bool = True,
        turn_id: Optional[str] = None,
        sequence_number: int = 0
    ) -> int:
        """
        Appends a streaming transcript turn to the transcripts table.
        """
        clean_speaker = str(speaker).lower()
        if clean_speaker not in ("caller", "gemini", "agent", "operator", "system"):
            clean_speaker = "gemini" if "gemini" in clean_speaker or "ai" in clean_speaker else "caller"

        clean_text = str(text or "")
        final_int = 1 if is_final else 0
        clean_turn_id = str(turn_id or "")

        async with self._lock:
            conn = await self._get_conn()
            try:
                cursor = await conn.execute(
                    """
                    INSERT INTO transcripts (call_id, speaker, turn_id, text, is_final, sequence_number)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (call_id, clean_speaker, clean_turn_id, clean_text, final_int, sequence_number)
                )
                await conn.commit()
                return cursor.lastrowid
            except Exception:
                await conn.rollback()
                raise

    async def save_message(
        self,
        call_id: Optional[str],
        caller_name: Optional[str],
        phone_number: Optional[str],
        message: Optional[str],
        urgency: Optional[str] = "normal"
    ) -> int:
        """
        Stores a caller-left message with urgency and read tracking.
        Coerces empty foreign keys to None to prevent constraint failures.
        """
        clean_call_id = call_id if call_id else None
        clean_name = str(caller_name or "").strip() or "Unknown"
        clean_phone = str(phone_number or "").strip()
        clean_msg = str(message or "")

        clean_urgency = str(urgency or "normal").lower().strip()
        if clean_urgency not in ("low", "normal", "medium", "high", "critical"):
            clean_urgency = "normal"

        async with self._lock:
            conn = await self._get_conn()
            try:
                cursor = await conn.execute(
                    """
                    INSERT INTO messages (call_id, caller_name, phone_number, message, urgency)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (clean_call_id, clean_name, clean_phone, clean_msg, clean_urgency)
                )
                await conn.commit()
                return cursor.lastrowid
            except Exception:
                await conn.rollback()
                raise

    async def save_summary(
        self,
        call_id: str,
        summary: str,
        action_items: Optional[Union[List[str], str]] = None,
        sentiment: Optional[str] = None,
        key_topics: Optional[Union[List[str], str]] = None
    ) -> int:
        """
        Persists a post-call summary using UPSERT on call_id conflict.
        Serializes action_items and key_topics lists into JSON strings.
        """
        if isinstance(action_items, list):
            action_items_str = json.dumps(action_items)
        elif isinstance(action_items, str):
            action_items_str = json.dumps([action_items])
        else:
            action_items_str = "[]"

        if isinstance(key_topics, list):
            key_topics_str = json.dumps(key_topics)
        elif isinstance(key_topics, str):
            key_topics_str = json.dumps([key_topics])
        else:
            key_topics_str = "[]"

        clean_sentiment = str(sentiment or "neutral")

        async with self._lock:
            conn = await self._get_conn()
            try:
                cursor = await conn.execute(
                    """
                    INSERT INTO summaries (call_id, summary, action_items, sentiment, key_topics)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(call_id) DO UPDATE SET
                        summary = excluded.summary,
                        action_items = excluded.action_items,
                        sentiment = excluded.sentiment,
                        key_topics = excluded.key_topics,
                        created_at = CURRENT_TIMESTAMP
                    """,
                    (call_id, summary, action_items_str, clean_sentiment, key_topics_str)
                )
                await conn.commit()
                return cursor.lastrowid
            except Exception:
                await conn.rollback()
                raise

    async def get_recent_calls(
        self,
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Queries call records ordered by started_at DESC with pagination and filtering.
        """
        limit = max(0, limit)
        offset = max(0, offset)

        conn = await self._get_conn()
        query = "SELECT * FROM calls"
        params: List[Any] = []

        if status:
            query += " WHERE status = ?"
            params.append(status.lower())

        query += " ORDER BY started_at DESC, rowid DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        async with conn.execute(query, tuple(params)) as cur:
            rows = await cur.fetchall()
            return [dict(row) for row in rows]

    async def get_call_details(self, call_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves complete call details including transcripts, messages, and summary.
        """
        conn = await self._get_conn()

        async with conn.execute("SELECT * FROM calls WHERE id = ?", (call_id,)) as cur:
            call_row = await cur.fetchone()
            if not call_row:
                return None
            call = dict(call_row)

        async with conn.execute(
            "SELECT * FROM transcripts WHERE call_id = ? ORDER BY id ASC",
            (call_id,)
        ) as cur:
            trans_rows = await cur.fetchall()
            call["transcripts"] = [dict(r) for r in trans_rows]

        async with conn.execute(
            "SELECT * FROM messages WHERE call_id = ? ORDER BY id ASC",
            (call_id,)
        ) as cur:
            msg_rows = await cur.fetchall()
            call["messages"] = [dict(r) for r in msg_rows]

        async with conn.execute(
            "SELECT summary, action_items, sentiment, key_topics, created_at FROM summaries WHERE call_id = ?",
            (call_id,)
        ) as cur:
            summary_row = await cur.fetchone()
            if summary_row:
                s_dict = dict(summary_row)
                try:
                    s_dict["action_items"] = json.loads(s_dict["action_items"])
                except Exception:
                    s_dict["action_items"] = []
                try:
                    s_dict["key_topics"] = json.loads(s_dict["key_topics"])
                except Exception:
                    s_dict["key_topics"] = []
                call["summary"] = s_dict
            else:
                call["summary"] = None

        return call

    async def get_transcripts(self, call_id: str) -> List[Dict[str, Any]]:
        """Retrieves chronological transcripts for a given call_id."""
        conn = await self._get_conn()
        async with conn.execute(
            "SELECT * FROM transcripts WHERE call_id = ? ORDER BY id ASC",
            (call_id,)
        ) as cur:
            rows = await cur.fetchall()
            return [dict(row) for row in rows]

    async def get_messages(
        self,
        unread_only: bool = False,
        urgency: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Queries recorded messages with optional unread and urgency filtering.
        """
        conn = await self._get_conn()
        query = "SELECT * FROM messages WHERE 1=1"
        params: List[Any] = []

        if unread_only:
            query += " AND is_read = 0"

        if urgency:
            query += " AND urgency = ?"
            params.append(urgency.lower())

        query += " ORDER BY created_at DESC, id DESC"
        async with conn.execute(query, tuple(params)) as cur:
            rows = await cur.fetchall()
            return [dict(row) for row in rows]

    async def update_message(
        self,
        message_id: int,
        is_read: Optional[bool] = None,
        action_taken: Optional[str] = None
    ) -> bool:
        """
        Updates read status and operator action note on a message.
        """
        if is_read is None and action_taken is None:
            return True

        updates = []
        params: List[Any] = []

        if is_read is not None:
            updates.append("is_read = ?")
            params.append(1 if is_read else 0)

        if action_taken is not None:
            updates.append("action_taken = ?")
            params.append(str(action_taken))

        params.append(message_id)
        query = f"UPDATE messages SET {', '.join(updates)} WHERE id = ?"

        async with self._lock:
            conn = await self._get_conn()
            try:
                cursor = await conn.execute(query, tuple(params))
                await conn.commit()
                return cursor.rowcount > 0
            except Exception:
                await conn.rollback()
                raise

    async def delete_call(self, call_id: str) -> bool:
        """
        Deletes a call record. Relational cascading deletes automatically remove
        associated transcripts, messages, and summary.
        """
        async with self._lock:
            conn = await self._get_conn()
            try:
                cursor = await conn.execute("DELETE FROM calls WHERE id = ?", (call_id,))
                await conn.commit()
                return cursor.rowcount > 0
            except Exception:
                await conn.rollback()
                raise
