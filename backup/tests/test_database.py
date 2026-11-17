"""
Comprehensive Asynchronous Unit and Concurrency Test Suite for server.database.
Covers SQLite WAL mode, PRAGMAs, CRUD operations, foreign key cascades,
performance index query plans, unicode edge cases, and high concurrency.
"""

import asyncio
import os
import sqlite3
import tempfile
import uuid
from typing import AsyncGenerator

import aiosqlite
import pytest
import pytest_asyncio

from server.database import DatabaseManager


@pytest_asyncio.fixture
async def db_manager() -> AsyncGenerator[DatabaseManager, None]:
    """Provides an isolated DatabaseManager instance backed by a temporary file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    manager = DatabaseManager(path)
    await manager.init_db()

    yield manager

    await manager.close()
    for ext in ["", "-wal", "-shm"]:
        p = path + ext
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass


# ============================================================================
# 1. Database Lifecycle & PRAGMA Verification Tests
# ============================================================================

class TestDatabaseLifecycleAndPRAGMAs:
    """Verifies connection lifecycle, schema initialization, and PRAGMAs."""

    @pytest.mark.asyncio
    async def test_init_db_creates_tables_and_pragmas(self, db_manager: DatabaseManager):
        """Verify journal_mode is WAL, foreign_keys is ON, and busy_timeout is 5000."""
        conn = await db_manager._get_conn()

        async with conn.execute("PRAGMA journal_mode") as cur:
            row = await cur.fetchone()
            assert row[0].lower() == "wal"

        async with conn.execute("PRAGMA foreign_keys") as cur:
            row = await cur.fetchone()
            assert row[0] == 1

        async with conn.execute("PRAGMA busy_timeout") as cur:
            row = await cur.fetchone()
            assert row[0] == 5000

        async with conn.execute("PRAGMA synchronous") as cur:
            row = await cur.fetchone()
            assert row[0] in (1, "NORMAL", "normal")

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Verify DatabaseManager functions cleanly as an async context manager."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)

        async with DatabaseManager(path) as db:
            call_id = await db.create_call({"caller_id": "+15551112222"})
            assert call_id.startswith("call_")
            details = await db.get_call_details(call_id)
            assert details["caller_id"] == "+15551112222"

        # After exiting context, file can be removed cleanly
        os.remove(path)

    @pytest.mark.asyncio
    async def test_parent_directory_creation(self):
        """Verify init_db creates nested parent directories automatically."""
        tmp_dir = tempfile.mkdtemp()
        nested_path = os.path.join(tmp_dir, "nested", "subfolder", "test.db")

        mgr = DatabaseManager(nested_path)
        await mgr.init_db()
        assert os.path.exists(nested_path)
        await mgr.close()

        # Cleanup
        for root, dirs, files in os.walk(tmp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(tmp_dir)


# ============================================================================
# 2. Calls Table CRUD Tests
# ============================================================================

class TestCallsCRUD:
    """Verifies calls table creation, status updates, pagination, and deletion."""

    @pytest.mark.asyncio
    async def test_create_call_defaults(self, db_manager: DatabaseManager):
        """Verify create_call sets sensible defaults for optional fields."""
        call_id = await db_manager.create_call({"caller_id": "+15553334444"})
        assert call_id is not None

        details = await db_manager.get_call_details(call_id)
        assert details["id"] == call_id
        assert details["caller_id"] == "+15553334444"
        assert details["direction"] == "inbound"
        assert details["status"] == "ringing"
        assert details["driver_type"] == "virtual"
        assert details["approval_action"] == "pending"
        assert details["duration"] == 0.0

    @pytest.mark.asyncio
    async def test_create_call_custom_fields(self, db_manager: DatabaseManager):
        """Verify create_call with all custom fields specified."""
        custom_id = f"call_custom_{uuid.uuid4().hex[:6]}"
        call_id = await db_manager.create_call({
            "id": custom_id,
            "direction": "outbound",
            "caller_id": "+15550001111",
            "callee_id": "+15552223333",
            "caller_name": "Test Agency",
            "status": "active",
            "approval_action": "ai_takeover",
            "driver_type": "windows",
            "goal_prompt": "Screen incoming business inquiry."
        })
        assert call_id == custom_id

        details = await db_manager.get_call_details(call_id)
        assert details["direction"] == "outbound"
        assert details["caller_name"] == "Test Agency"
        assert details["goal_prompt"] == "Screen incoming business inquiry."

    @pytest.mark.asyncio
    async def test_update_call_status_completed_metrics(self, db_manager: DatabaseManager):
        """Verify updating status to completed records ended_at, duration, and hangup_reason."""
        call_id = await db_manager.create_call({"caller_id": "+15555556666"})
        success = await db_manager.update_call_status(
            call_id,
            "completed",
            hangup_reason="normal_clearing",
            duration=32.4
        )
        assert success is True

        details = await db_manager.get_call_details(call_id)
        assert details["status"] == "completed"
        assert details["hangup_reason"] == "normal_clearing"
        assert details["end_reason"] == "normal_clearing"
        assert details["duration"] == 32.4
        assert details["duration_seconds"] == 32
        assert details["ended_at"] is not None

    @pytest.mark.asyncio
    async def test_update_call_status_nonexistent_returns_false(self, db_manager: DatabaseManager):
        """Updating status of nonexistent call returns False."""
        success = await db_manager.update_call_status("call_does_not_exist", "completed")
        assert success is False

    @pytest.mark.asyncio
    async def test_get_recent_calls_pagination_and_filter(self, db_manager: DatabaseManager):
        """Verify pagination with limit/offset and status filtering."""
        c1 = await db_manager.create_call({"caller_id": "1", "status": "ringing"})
        await asyncio.sleep(0.01)
        c2 = await db_manager.create_call({"caller_id": "2", "status": "active"})
        await asyncio.sleep(0.01)
        c3 = await db_manager.create_call({"caller_id": "3", "status": "completed"})

        # Pagination: limit=2
        page1 = await db_manager.get_recent_calls(limit=2, offset=0)
        assert len(page1) == 2
        assert page1[0]["id"] == c3  # Most recent first
        assert page1[1]["id"] == c2

        # Status filter
        active_calls = await db_manager.get_recent_calls(status="active")
        assert len(active_calls) == 1
        assert active_calls[0]["id"] == c2

    @pytest.mark.asyncio
    async def test_delete_call(self, db_manager: DatabaseManager):
        """Verify deleting an existing call returns True and removes record."""
        call_id = await db_manager.create_call({"caller_id": "+15559998888"})
        assert await db_manager.delete_call(call_id) is True
        assert await db_manager.get_call_details(call_id) is None
        assert await db_manager.delete_call(call_id) is False


# ============================================================================
# 3. Transcripts Table CRUD Tests
# ============================================================================

class TestTranscriptsCRUD:
    """Verifies transcripts storage, multi-speaker ordering, and foreign key integrity."""

    @pytest.mark.asyncio
    async def test_append_and_retrieve_transcripts(self, db_manager: DatabaseManager):
        """Verify transcripts append chronologically and are retrieved via get_transcripts."""
        call_id = await db_manager.create_call({"caller_id": "+15557778888"})

        t1 = await db_manager.append_transcript(call_id, "caller", "Hello, who is this?", is_final=True)
        t2 = await db_manager.append_transcript(call_id, "gemini", "Hello! I am Aoede, AI calling assistant.", is_final=True)
        t3 = await db_manager.append_transcript(call_id, "agent", "How can I direct your call?", is_final=True)

        assert t1 > 0 and t2 > t1 and t3 > t2

        transcripts = await db_manager.get_transcripts(call_id)
        assert len(transcripts) == 3
        assert transcripts[0]["speaker"] == "caller"
        assert transcripts[0]["text"] == "Hello, who is this?"
        assert transcripts[1]["speaker"] == "gemini"
        assert transcripts[2]["speaker"] == "agent"

    @pytest.mark.asyncio
    async def test_append_transcript_foreign_key_violation(self, db_manager: DatabaseManager):
        """Verify appending a transcript to a nonexistent call raises sqlite3.IntegrityError."""
        with pytest.raises(sqlite3.IntegrityError):
            await db_manager.append_transcript("nonexistent_call_id", "caller", "Test message")

    @pytest.mark.asyncio
    async def test_transcript_multilingual_unicode_and_emojis(self, db_manager: DatabaseManager):
        """Verify UTF-8 unicode and emojis persist and retrieve with exact fidelity."""
        call_id = await db_manager.create_call({"caller_id": "+15551239999"})
        unicode_payload = "📞 नमस्ते 你好 مرحبا Bonjour 🤖 AI screening active! ✨"

        await db_manager.append_transcript(call_id, "gemini", unicode_payload, is_final=True)
        transcripts = await db_manager.get_transcripts(call_id)
        assert len(transcripts) == 1
        assert transcripts[0]["text"] == unicode_payload

    @pytest.mark.asyncio
    async def test_transcript_massive_chunk_50k_characters(self, db_manager: DatabaseManager):
        """Verify large 50,000 character transcript strings persist accurately."""
        call_id = await db_manager.create_call({"caller_id": "+15554443333"})
        massive_text = "Voice Assistant Long Speech Chunk. " * 1500  # ~52,500 chars

        await db_manager.append_transcript(call_id, "gemini", massive_text, is_final=True)
        transcripts = await db_manager.get_transcripts(call_id)
        assert len(transcripts) == 1
        assert transcripts[0]["text"] == massive_text


# ============================================================================
# 4. Messages Table CRUD Tests
# ============================================================================

class TestMessagesCRUD:
    """Verifies message storage, null coercion, urgency filters, and updates."""

    @pytest.mark.asyncio
    async def test_save_and_get_messages(self, db_manager: DatabaseManager):
        """Verify saving messages and querying with unread and urgency filters."""
        call_id = await db_manager.create_call({"caller_id": "+15551113333"})

        m1 = await db_manager.save_message(call_id, "Alice Smith", "+15551113333", "Please call back regarding invoice #402", urgency="high")
        m2 = await db_manager.save_message(call_id, "Bob Jones", "+15552224444", "Just checking in.", urgency="low")

        assert m1 > 0 and m2 > m1

        all_msgs = await db_manager.get_messages()
        assert len(all_msgs) == 2

        high_msgs = await db_manager.get_messages(urgency="high")
        assert len(high_msgs) == 1
        assert high_msgs[0]["caller_name"] == "Alice Smith"

        unread_msgs = await db_manager.get_messages(unread_only=True)
        assert len(unread_msgs) == 2

    @pytest.mark.asyncio
    async def test_save_message_null_coercion(self, db_manager: DatabaseManager):
        """Verify null fields in save_message coerce cleanly to non-null defaults."""
        call_id = await db_manager.create_call({"caller_id": "+15558887777"})
        msg_id = await db_manager.save_message(call_id, None, None, "Null field test", urgency=None)
        assert msg_id > 0

        msgs = await db_manager.get_messages()
        saved = [m for m in msgs if m["id"] == msg_id][0]
        assert saved["caller_name"] == "Unknown"
        assert saved["phone_number"] == ""
        assert saved["urgency"] == "normal"

    @pytest.mark.asyncio
    async def test_update_message_is_read_and_action_taken(self, db_manager: DatabaseManager):
        """Verify updating is_read and action_taken notes on messages."""
        call_id = await db_manager.create_call({"caller_id": "+15559990000"})
        msg_id = await db_manager.save_message(call_id, "Charlie", "555-0100", "Contract review requested", urgency="medium")

        success = await db_manager.update_message(msg_id, is_read=True, action_taken="Returned call at 2:00 PM")
        assert success is True

        unread = await db_manager.get_messages(unread_only=True)
        assert not any(m["id"] == msg_id for m in unread)

        all_msgs = await db_manager.get_messages()
        target = [m for m in all_msgs if m["id"] == msg_id][0]
        assert target["is_read"] == 1
        assert target["action_taken"] == "Returned call at 2:00 PM"


# ============================================================================
# 5. Summaries Table CRUD & JSON Tests
# ============================================================================

class TestSummariesCRUD:
    """Verifies summaries persistence, JSON serialization, and UPSERT semantics."""

    @pytest.mark.asyncio
    async def test_save_summary_and_json_deserialization(self, db_manager: DatabaseManager):
        """Verify save_summary serializes lists and get_call_details returns Python lists."""
        call_id = await db_manager.create_call({"caller_id": "+15553331111"})
        summary_text = "Client called regarding annual software license renewal."
        action_items = ["Send updated pricing proposal", "Schedule demo for Thursday"]
        key_topics = ["Licensing", "Pricing", "Demo"]

        summary_id = await db_manager.save_summary(
            call_id=call_id,
            summary=summary_text,
            action_items=action_items,
            sentiment="positive",
            key_topics=key_topics
        )
        assert summary_id > 0

        details = await db_manager.get_call_details(call_id)
        assert details["summary"] is not None
        assert details["summary"]["summary"] == summary_text
        assert details["summary"]["action_items"] == action_items
        assert details["summary"]["key_topics"] == key_topics
        assert details["summary"]["sentiment"] == "positive"

    @pytest.mark.asyncio
    async def test_save_summary_upsert_on_conflict(self, db_manager: DatabaseManager):
        """Verify saving summary twice for same call_id updates record without error."""
        call_id = await db_manager.create_call({"caller_id": "+15554442222"})

        await db_manager.save_summary(call_id, "Initial summary", action_items=["Task 1"])
        await db_manager.save_summary(call_id, "Updated final summary", action_items=["Task 1", "Task 2"])

        details = await db_manager.get_call_details(call_id)
        assert details["summary"]["summary"] == "Updated final summary"
        assert details["summary"]["action_items"] == ["Task 1", "Task 2"]


# ============================================================================
# 6. Relational Integrity & Cascading Deletes Tests
# ============================================================================

class TestRelationalIntegrityAndCascades:
    """Verifies ON DELETE CASCADE removes all child records when a call is deleted."""

    @pytest.mark.asyncio
    async def test_cascading_delete_all_child_records(self, db_manager: DatabaseManager):
        """Deleting a call cascades and deletes transcripts, messages, and summaries."""
        call_id = await db_manager.create_call({"caller_id": "+15556667777"})

        await db_manager.append_transcript(call_id, "caller", "Turn 1")
        await db_manager.append_transcript(call_id, "gemini", "Turn 2")
        await db_manager.save_message(call_id, "Caller", "555-6667", "Msg 1")
        await db_manager.save_summary(call_id, "Summary 1", action_items=["Action 1"])

        conn = await db_manager._get_conn()
        # Verify counts before delete
        async with conn.execute("SELECT COUNT(*) FROM transcripts WHERE call_id = ?", (call_id,)) as cur:
            assert (await cur.fetchone())[0] == 2
        async with conn.execute("SELECT COUNT(*) FROM messages WHERE call_id = ?", (call_id,)) as cur:
            assert (await cur.fetchone())[0] == 1
        async with conn.execute("SELECT COUNT(*) FROM summaries WHERE call_id = ?", (call_id,)) as cur:
            assert (await cur.fetchone())[0] == 1

        # Delete call
        assert await db_manager.delete_call(call_id) is True

        # Verify all children were cascade deleted
        async with conn.execute("SELECT COUNT(*) FROM transcripts WHERE call_id = ?", (call_id,)) as cur:
            assert (await cur.fetchone())[0] == 0
        async with conn.execute("SELECT COUNT(*) FROM messages WHERE call_id = ?", (call_id,)) as cur:
            assert (await cur.fetchone())[0] == 0
        async with conn.execute("SELECT COUNT(*) FROM summaries WHERE call_id = ?", (call_id,)) as cur:
            assert (await cur.fetchone())[0] == 0


# ============================================================================
# 7. Database Performance Indexes Tests
# ============================================================================

class TestDatabaseIndexes:
    """Verifies SQLite query planner utilizes defined performance indexes."""

    @pytest.mark.asyncio
    async def test_explain_query_plan_uses_indexes(self, db_manager: DatabaseManager):
        """Verify EXPLAIN QUERY PLAN demonstrates index usage for key query patterns."""
        conn = await db_manager._get_conn()

        # idx_calls_started_at
        async with conn.execute("EXPLAIN QUERY PLAN SELECT * FROM calls ORDER BY started_at DESC LIMIT 10") as cur:
            rows = await cur.fetchall()
            plan = " ".join([r[3] for r in rows])
            assert "idx_calls_started_at" in plan

        # idx_transcripts_call_id
        async with conn.execute("EXPLAIN QUERY PLAN SELECT * FROM transcripts WHERE call_id = ?", ("dummy_id",)) as cur:
            rows = await cur.fetchall()
            plan = " ".join([r[3] for r in rows])
            assert "idx_transcripts_call_id" in plan

        # idx_messages_call_id
        async with conn.execute("EXPLAIN QUERY PLAN SELECT * FROM messages WHERE call_id = ?", ("dummy_id",)) as cur:
            rows = await cur.fetchall()
            plan = " ".join([r[3] for r in rows])
            assert "idx_messages_call_id" in plan

        # idx_summaries_call_id
        async with conn.execute("EXPLAIN QUERY PLAN SELECT * FROM summaries WHERE call_id = ?", ("dummy_id",)) as cur:
            rows = await cur.fetchall()
            plan = " ".join([r[3] for r in rows])
            assert "idx_summaries_call_id" in plan or "sqlite_autoindex_summaries" in plan


# ============================================================================
# 8. High Concurrency & Contention Tests
# ============================================================================

class TestHighConcurrencyAndContention:
    """Verifies that high concurrency read/write workloads run without locking errors."""

    @pytest.mark.asyncio
    async def test_concurrent_writes_and_reads_under_wal(self, db_manager: DatabaseManager):
        """50 concurrent writes and 50 concurrent reads execute without database locked error."""
        call_id = await db_manager.create_call({"caller_id": "+15550009999"})

        async def worker_write(idx: int):
            await db_manager.append_transcript(call_id, "caller", f"Concurrent turn {idx}")
            await db_manager.save_message(call_id, f"Caller {idx}", f"555-{idx:04d}", f"Msg {idx}")

        async def worker_read():
            await db_manager.get_recent_calls(limit=10)
            await db_manager.get_transcripts(call_id)
            await db_manager.get_messages(unread_only=True)

        tasks = [worker_write(i) for i in range(50)] + [worker_read() for _ in range(50)]
        import random
        random.shuffle(tasks)

        await asyncio.gather(*tasks)

        transcripts = await db_manager.get_transcripts(call_id)
        assert len(transcripts) == 50
        messages = await db_manager.get_messages()
        assert len(messages) >= 50
