"""
Centralized Configuration for AI Voice Calling Agent.
Provides AgentConfig (Pydantic model) with all tuneable parameters:
model, voice persona, audio rates, telephony driver, inbound policy, and system prompt.
"""

import os
from typing import Optional, List
from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Top-level configuration for the AI Voice Calling Agent."""

    # --- Gemini Live 3.8 ---
    model_name: str = Field(
        default="gemini-3.8-live",
        description="Gemini Live model identifier."
    )
    voice_name: str = Field(
        default="Aoede",
        description="Voice persona: Puck, Charon, Aoede, Fenrir, Kore."
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Gemini API key. Falls back to GEMINI_API_KEY env var."
    )

    # --- Audio ---
    input_sample_rate: int = Field(
        default=16000,
        description="PCM input sample rate for Gemini Live (Hz)."
    )
    output_sample_rate: int = Field(
        default=24000,
        description="PCM output sample rate from Gemini Live (Hz)."
    )

    # --- Telephony ---
    telephony_driver: str = Field(
        default="virtual",
        description="Telephony driver: 'virtual', 'windows', or 'linux'."
    )
    input_device: Optional[str] = Field(
        default=None,
        description="Audio input device name or index override."
    )
    output_device: Optional[str] = Field(
        default=None,
        description="Audio output device name or index override."
    )

    # --- Server ---
    host: str = Field(default="0.0.0.0", description="Server bind address.")
    port: int = Field(default=8000, description="Server port.")

    # --- Inbound Call Policy ---
    inbound_policy: str = Field(
        default="manual_approval",
        description="Inbound call policy: 'manual_approval' or 'auto_answer'."
    )
    auto_answer_delay_seconds: int = Field(
        default=4,
        description="Seconds to wait before auto-answering (if auto_answer policy)."
    )

    # --- System Prompt ---
    system_prompt: str = Field(
        default=(
            "You are an AI call screening assistant answering phone calls on behalf of the user. "
            "You speak with a warm, soft-spoken, and professional female voice (Aoede). "
            "Your role is to politely greet the caller, inquire about the purpose of their call, "
            "take messages if the user is unavailable using the take_message tool, "
            "transfer important calls to a human using transfer_to_human, or gracefully end the call using end_call. "
            "Be concise, courteous, and natural as in a real phone conversation."
        ),
        description="System instruction for Gemini Live session."
    )

    @classmethod
    def from_env(cls, **overrides) -> "AgentConfig":
        """
        Creates config from environment variables with optional overrides.
        Env vars: GEMINI_API_KEY, VOICE_CALLER_DRIVER, VOICE_CALLER_PORT,
                  VOICE_CALLER_VOICE, VOICE_CALLER_HOST
        """
        env_map = {
            "api_key": os.environ.get("GEMINI_API_KEY"),
            "telephony_driver": os.environ.get("VOICE_CALLER_DRIVER", "virtual"),
            "port": int(os.environ.get("VOICE_CALLER_PORT", "8000")),
            "host": os.environ.get("VOICE_CALLER_HOST", "0.0.0.0"),
            "voice_name": os.environ.get("VOICE_CALLER_VOICE", "Aoede"),
        }
        # Remove None values so defaults apply
        env_map = {k: v for k, v in env_map.items() if v is not None}
        env_map.update(overrides)
        return cls(**env_map)
