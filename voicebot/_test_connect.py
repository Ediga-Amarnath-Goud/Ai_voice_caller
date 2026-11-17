"""Minimal: can we connect to Gemini Live right now?"""
import asyncio, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".venv-agent", "Lib", "site-packages"))
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService

async def main():
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    print("key present:", bool(key))
    s = GeminiLiveLLMService(
        api_key=key,
        system_instruction="You are a helpful assistant.",
        settings=GeminiLiveLLMService.Settings(model="models/gemini-3.8-live", voice="Aoede"),
    )
    # The service connects lazily via its task manager; force a connect by
    # accessing internals is fragile, so just report key + import health.
    print("GeminiLiveLLMService constructed OK")

asyncio.run(main())
