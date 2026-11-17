"""
Comprehensive unit tests for GeminiLiveClient and core/tools.py.
Verifies tool declarations, registry handlers, live session connection lifecycle,
audio streaming, whisper injection, audio/transcript receipt, barge-in detection,
tool execution handshake, and cancellation cleanup.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from types import SimpleNamespace
from google.genai import types

from core.tools import get_tool_declarations, ToolRegistry
from core.gemini_live import GeminiLiveClient, DEFAULT_SYSTEM_INSTRUCTION


# ============================================================================
# Tool Declarations and Registry Unit Tests
# ============================================================================

class TestToolDeclarationsAndRegistry:
    """Tests schema declarations and execution registry for tool calling."""

    def test_get_tool_declarations_schemas(self):
        """Verifies all 4 required tools are properly declared with correct schemas."""
        tools = get_tool_declarations()
        assert len(tools) == 1
        tool = tools[0]
        declarations = {fn.name: fn for fn in tool.function_declarations}

        assert set(declarations.keys()) == {
            "take_message",
            "end_call",
            "transfer_to_human",
            "log_call_summary",
        }

        # Validate take_message
        take_msg = declarations["take_message"]
        assert "caller_name" in take_msg.parameters.properties
        assert "message" in take_msg.parameters.properties
        assert "urgency" in take_msg.parameters.properties
        assert set(take_msg.parameters.required) == {"caller_name", "message"}

        # Validate end_call
        end_call = declarations["end_call"]
        assert "reason" in end_call.parameters.properties
        assert end_call.parameters.required == ["reason"]

        # Validate transfer_to_human
        transfer = declarations["transfer_to_human"]
        assert "reason" in transfer.parameters.properties
        assert transfer.parameters.required == ["reason"]

        # Validate log_call_summary
        summary = declarations["log_call_summary"]
        assert "summary" in summary.parameters.properties
        assert "action_items" in summary.parameters.properties
        assert summary.parameters.required == ["summary"]

    @pytest.mark.asyncio
    async def test_default_tool_handlers(self):
        """Verifies default implementations return valid structured dictionaries."""
        registry = ToolRegistry()

        # 1. take_message
        res1 = await registry.execute_tool(
            "take_message",
            {"caller_name": "Alice Smith", "phone_number": "555-0199", "message": "Please call back", "urgency": "high"}
        )
        assert res1["status"] == "saved"
        assert res1["caller_name"] == "Alice Smith"
        assert res1["phone_number"] == "555-0199"
        assert res1["urgency"] == "high"

        # 2. end_call
        res2 = await registry.execute_tool("end_call", {"reason": "conversation_finished"})
        assert res2["status"] == "call_ended"
        assert res2["reason"] == "conversation_finished"

        # 3. transfer_to_human
        res3 = await registry.execute_tool("transfer_to_human", {"reason": "caller_angry"})
        assert res3["status"] == "transfer_initiated"
        assert res3["on_hold"] is True
        assert res3["reason"] == "caller_angry"

        # 4. log_call_summary
        res4 = await registry.execute_tool(
            "log_call_summary",
            {"summary": "Inquiry about pricing", "action_items": ["Send PDF quote", "Follow up tomorrow"]}
        )
        assert res4["status"] == "logged"
        assert res4["summary"] == "Inquiry about pricing"
        assert res4["action_item_count"] == 2
        assert res4["action_items"] == ["Send PDF quote", "Follow up tomorrow"]

    @pytest.mark.asyncio
    async def test_custom_handler_registration_and_dispatch(self):
        """Tests custom synchronous and asynchronous handler overrides."""
        registry = ToolRegistry()

        # Custom async handler
        async def custom_take_message(caller_name: str, message: str, **kwargs):
            return {"status": "custom_saved", "name": caller_name, "body": message}

        registry.register_handler("take_message", custom_take_message)
        res = await registry.execute_tool("take_message", {"caller_name": "Dr. House", "message": "Lupus check"})
        assert res["status"] == "custom_saved"
        assert res["name"] == "Dr. House"

        # Custom sync handler
        def custom_sync(reason: str):
            return {"status": "sync_ended", "r": reason}

        registry.register_handler("end_call", custom_sync)
        res_sync = await registry.execute_tool("end_call", {"reason": "test"})
        assert res_sync["status"] == "sync_ended"

    @pytest.mark.asyncio
    async def test_unregistered_tool_and_exception_handling(self):
        """Tests proper error dictionary return on missing tool or handler exceptions."""
        registry = ToolRegistry()

        # Unknown tool
        res_unknown = await registry.execute_tool("unknown_tool", {"foo": "bar"})
        assert res_unknown["status"] == "error"
        assert "not found" in res_unknown["error"]

        # Failing tool
        def broken_handler(**kwargs):
            raise RuntimeError("Database connection lost")

        registry.register_handler("failing_tool", broken_handler)
        res_fail = await registry.execute_tool("failing_tool", {})
        assert res_fail["status"] == "error"
        assert "Database connection lost" in res_fail["error"]

    @pytest.mark.asyncio
    async def test_tool_execution_with_non_dict_arguments(self):
        """Verifies execute_tool gracefully returns error dict when given non-dict arguments."""
        registry = ToolRegistry()
        for bad_arg in [123, "string_payload", [1, 2, 3]]:
            res = await registry.execute_tool("take_message", bad_arg)
            assert isinstance(res, dict)
            assert res.get("status") == "error"
            assert f"Invalid arguments payload: expected dictionary, got {type(bad_arg).__name__}" in res.get("error", "")

        # Test optional default parameters dispatch cleanly with empty dict
        def tool_with_default(reason: str = "default_reason"):
            return {"status": "ok", "reason": reason}

        registry.register_handler("test_default_tool", tool_with_default)
        res_default = await registry.execute_tool("test_default_tool", {})
        assert res_default["status"] == "ok"
        assert res_default["reason"] == "default_reason"


# ============================================================================
# GeminiLiveClient Configuration and Initialization Tests
# ============================================================================

class TestGeminiLiveClientInitialization:
    """Tests client construction, default configurations, and parameters."""

    def test_default_initialization(self):
        """Verifies default parameters: model, voice, system prompt, and tools."""
        client = GeminiLiveClient(api_key="test-api-key")
        assert client.voice_name == "Aoede"
        assert client.model == "gemini-3.8-live"
        assert client.system_instruction == DEFAULT_SYSTEM_INSTRUCTION
        assert len(client.tools) == 1
        assert client.is_connected is False

    def test_custom_initialization(self):
        """Verifies custom voice, system prompt, and injected genai.Client."""
        custom_mock = MagicMock()
        client = GeminiLiveClient(
            api_key="custom-key",
            voice_name="Puck",
            system_instruction="Custom assistant prompt",
            model="gemini-3.8-live-custom",
            client=custom_mock,
        )
        assert client.voice_name == "Puck"
        assert client.system_instruction == "Custom assistant prompt"
        assert client.model == "gemini-3.8-live-custom"
        assert client.client is custom_mock

    def test_config_builder(self):
        """Verifies that _build_config creates a valid LiveConnectConfig."""
        client = GeminiLiveClient(api_key="dummy-key", voice_name="Aoede")
        config = client._build_config()

        assert config.speech_config.voice_config.prebuilt_voice_config.voice_name == "Aoede"
        assert len(config.tools) == 1
        assert config.system_instruction is not None


# ============================================================================
# GeminiLiveClient Live Session Mock Tests
# ============================================================================

class AsyncIter:
    """Helper to simulate async iteration over a sequence of messages."""
    def __init__(self, items):
        self.items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.items:
            # Sleep briefly to keep loop alive until cancelled if needed
            await asyncio.sleep(0.01)
            raise StopAsyncIteration
        return self.items.pop(0)


@pytest.fixture
def mock_session():
    """Returns a mock AsyncSession with async methods."""
    session = AsyncMock()
    session.send_realtime_input = AsyncMock()
    session.send_tool_response = AsyncMock()
    session.receive = MagicMock(return_value=AsyncIter([]))
    return session


@pytest.fixture
def mock_genai_client(mock_session):
    """Returns a mock genai.Client managing the session context manager."""
    client = MagicMock()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_session)
    cm.__aexit__ = AsyncMock(return_value=None)
    client.aio.live.connect = MagicMock(return_value=cm)
    return client


class TestGeminiLiveClientSession:
    """Tests connection lifecycle, input streaming, and server event processing."""

    @pytest.mark.asyncio
    async def test_connect_and_disconnect(self, mock_genai_client, mock_session):
        """Tests clean connection establishment and graceful disconnection."""
        client = GeminiLiveClient(api_key="test-key", client=mock_genai_client)

        assert client.is_connected is False
        await client.connect()
        assert client.is_connected is True
        mock_genai_client.aio.live.connect.assert_called_once()

        # Connect again while already connected should be idempotent
        await client.connect()
        assert client.is_connected is True

        # Disconnect
        await client.disconnect()
        assert client.is_connected is False

        # Disconnect again should be idempotent
        await client.disconnect()
        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_send_audio_chunk(self, mock_genai_client, mock_session):
        """Tests streaming 16kHz PCM audio chunk to send_realtime_input."""
        client = GeminiLiveClient(api_key="test-key", client=mock_genai_client)
        pcm_16k_data = b"\x00\x10\x20\x30" * 80  # 320 bytes

        # Before connect: should safely ignore without crashing
        await client.send_audio_chunk(pcm_16k_data)
        mock_session.send_realtime_input.assert_not_called()

        await client.connect()
        await client.send_audio_chunk(pcm_16k_data)

        mock_session.send_realtime_input.assert_awaited_once()
        call_kwargs = mock_session.send_realtime_input.call_args.kwargs
        assert "media" in call_kwargs
        blob = call_kwargs["media"]
        assert blob.data == pcm_16k_data
        assert blob.mime_type == "audio/pcm;rate=16000"

        # Empty chunk should not be sent
        mock_session.send_realtime_input.reset_mock()
        await client.send_audio_chunk(b"")
        mock_session.send_realtime_input.assert_not_called()

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_send_whisper(self, mock_genai_client, mock_session):
        """Tests sending silent operator whisper instruction via send_realtime_input."""
        client = GeminiLiveClient(api_key="test-key", client=mock_genai_client)

        # Before connect
        await client.send_whisper("Remind caller to pay overdue invoice")
        mock_session.send_realtime_input.assert_not_called()

        await client.connect()
        await client.send_whisper("Remind caller to pay overdue invoice")

        mock_session.send_realtime_input.assert_awaited_once_with(
            text="[OPERATOR INSTRUCTION: Remind caller to pay overdue invoice]"
        )

        # Empty whisper ignored
        mock_session.send_realtime_input.reset_mock()
        await client.send_whisper("")
        mock_session.send_realtime_input.assert_not_called()

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_receive_audio_and_transcripts(self, mock_genai_client, mock_session):
        """Tests receipt of model turn containing transcript tokens and 24kHz PCM audio."""
        # Setup simulated server messages
        part1 = MagicMock(text="Hello there, ", inline_data=None)
        part2 = MagicMock(
            text="how may I assist you?",
            inline_data=MagicMock(data=b"\x11\x22\x33\x44" * 100)
        )
        sc1 = MagicMock(
            interrupted=False,
            turn_complete=False,
            model_turn=MagicMock(parts=[part1, part2])
        )
        msg1 = MagicMock(server_content=sc1, tool_call=None)

        # Final turn completion message
        sc2 = MagicMock(interrupted=False, turn_complete=True, model_turn=None)
        msg2 = MagicMock(server_content=sc2, tool_call=None)

        mock_session.receive = MagicMock(return_value=AsyncIter([msg1, msg2]))

        received_transcripts = []
        received_audio_chunks = []

        async def transcript_callback(speaker: str, text: str, is_final: bool):
            received_transcripts.append((speaker, text, is_final))

        async def audio_callback(pcm_bytes: bytes):
            received_audio_chunks.append(pcm_bytes)

        client = GeminiLiveClient(api_key="test-key", client=mock_genai_client)
        client.set_on_transcript(transcript_callback)
        client.set_on_audio_received(audio_callback)

        await client.connect()
        await asyncio.sleep(0.05)  # Allow receive loop to process messages

        assert len(received_transcripts) == 3
        assert received_transcripts[0] == ("agent", "Hello there, ", False)
        assert received_transcripts[1] == ("agent", "how may I assist you?", False)
        assert received_transcripts[2] == ("agent", "", True)

        assert len(received_audio_chunks) == 1
        assert received_audio_chunks[0] == b"\x11\x22\x33\x44" * 100

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_barge_in_detection(self, mock_genai_client, mock_session):
        """Tests that server_content.interrupted triggers the on_interrupted callback."""
        sc_interrupted = MagicMock(
            interrupted=True,
            turn_complete=False,
            model_turn=None
        )
        msg = MagicMock(server_content=sc_interrupted, tool_call=None)
        mock_session.receive = MagicMock(return_value=AsyncIter([msg]))

        interrupted_called = asyncio.Event()

        def interrupted_callback():
            interrupted_called.set()

        client = GeminiLiveClient(api_key="test-key", client=mock_genai_client)
        client.set_on_interrupted(interrupted_callback)

        await client.connect()
        await asyncio.wait_for(interrupted_called.wait(), timeout=1.0)
        assert interrupted_called.is_set()

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_tool_call_handshake(self, mock_genai_client, mock_session):
        """Tests that Gemini Live tool_call executes the handler and sends back tool responses."""
        fc = SimpleNamespace(
            id="call_tx_999",
            name="take_message",
            args={"caller_name": "Samantha", "phone_number": "555-8888", "message": "URGENT meeting change", "urgency": "high"}
        )
        tool_call_obj = SimpleNamespace(function_calls=[fc])
        msg = SimpleNamespace(server_content=None, tool_call=tool_call_obj)

        mock_session.receive = MagicMock(return_value=AsyncIter([msg]))

        client = GeminiLiveClient(api_key="test-key", client=mock_genai_client)

        await client.connect()
        await asyncio.sleep(0.05)  # Allow receive loop to execute tool

        mock_session.send_tool_response.assert_awaited_once()
        call_kwargs = mock_session.send_tool_response.call_args.kwargs
        assert "function_responses" in call_kwargs
        responses = call_kwargs["function_responses"]
        assert len(responses) == 1

        fr = responses[0]
        assert fr.id == "call_tx_999"
        assert fr.name == "take_message"
        assert fr.response["result"]["status"] == "saved"
        assert fr.response["result"]["caller_name"] == "Samantha"

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_receive_loop_cancellation_and_error_resilience(self, mock_genai_client, mock_session):
        """Tests that exceptions in receive loop are handled gracefully and disconnect completes cleanly."""
        async def faulty_generator():
            raise RuntimeError("Connection dropped by peer")
            yield  # Unreachable

        mock_session.receive = MagicMock(return_value=faulty_generator())

        client = GeminiLiveClient(api_key="test-key", client=mock_genai_client)
        await client.connect()
        await asyncio.sleep(0.05)

        # Client disconnects cleanly without raising unhandled exception
        await client.disconnect()
        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_receive_loop_network_drop_and_reconnect(self, mock_genai_client, mock_session):
        """Verifies that on network connection drop, is_connected becomes False and client can reconnect."""
        async def dropping_generator():
            await asyncio.sleep(0.01)
            raise ConnectionResetError("Connection reset by peer")
            yield  # Unreachable

        mock_session.receive = MagicMock(return_value=dropping_generator())

        client = GeminiLiveClient(api_key="test-key", client=mock_genai_client)
        await client.connect()
        assert client.is_connected is True

        # Wait for the receive loop to encounter the error and terminate
        await asyncio.sleep(0.05)

        # Confirm client is no longer connected and session is cleared
        assert client.is_connected is False
        assert client._session is None

        # Reset mock receive with normal iterator for reconnect
        mock_session.receive = MagicMock(return_value=AsyncIter([]))

        # Reconnect should succeed without being blocked by "already connected"
        await client.connect()
        assert client.is_connected is True
        assert mock_genai_client.aio.live.connect.call_count == 2

        await client.disconnect()
        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_disconnect_clears_already_done_task(self, mock_genai_client, mock_session):
        """Verifies that disconnect() clears self._receive_task even if the task is already done."""
        # Empty iterator completes immediately so task finishes
        mock_session.receive = MagicMock(return_value=AsyncIter([]))

        client = GeminiLiveClient(api_key="test-key", client=mock_genai_client)
        await client.connect()
        await asyncio.sleep(0.05)
        assert client._receive_task is not None
        assert client._receive_task.done() is True

        await client.disconnect()
        assert client._receive_task is None
        assert client.is_connected is False
