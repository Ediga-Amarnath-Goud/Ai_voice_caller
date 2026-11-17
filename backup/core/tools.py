"""
Function Calling Tools and Registry for Gemini Live 3.8.
Defines Google GenAI Tool schemas and provides a tool registry with default handlers
for take_message, end_call, transfer_to_human, and log_call_summary.
"""

import inspect
import logging
from typing import Callable, Awaitable, Dict, Any, List, Optional, Union
from google.genai import types

logger = logging.getLogger("core.tools")


def get_tool_declarations() -> List[types.Tool]:
    """
    Returns the list of google.genai types.Tool definitions for Gemini Live 3.8.
    Declares:
        1. take_message
        2. end_call
        3. transfer_to_human
        4. log_call_summary
    """
    return [
        types.Tool(
            function_declarations=[
                # 1. take_message
                types.FunctionDeclaration(
                    name="take_message",
                    description=(
                        "Save a structured message from the caller to the database when the owner is unavailable."
                    ),
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "caller_name": types.Schema(
                                type=types.Type.STRING,
                                description="Full name or company name of the caller."
                            ),
                            "phone_number": types.Schema(
                                type=types.Type.STRING,
                                description="Callback telephone number provided by the caller."
                            ),
                            "message": types.Schema(
                                type=types.Type.STRING,
                                description="Detailed message text explaining the purpose of the call."
                            ),
                            "urgency": types.Schema(
                                type=types.Type.STRING,
                                description="Urgency level of the message.",
                                enum=["low", "medium", "high"]
                            )
                        },
                        required=["caller_name", "message"]
                    )
                ),
                # 2. end_call
                types.FunctionDeclaration(
                    name="end_call",
                    description=(
                        "Hang up and terminate the telephone call when the conversation is finished, "
                        "after taking a message, or declining spam."
                    ),
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "reason": types.Schema(
                                type=types.Type.STRING,
                                description=(
                                    "Reason for ending the call (e.g. 'conversation_finished', "
                                    "'message_taken', 'spam_declined', 'caller_hung_up')."
                                )
                            )
                        },
                        required=["reason"]
                    )
                ),
                # 3. transfer_to_human
                types.FunctionDeclaration(
                    name="transfer_to_human",
                    description="Place the caller on hold and request an immediate pickup by the human operator.",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "reason": types.Schema(
                                type=types.Type.STRING,
                                description="Reason why the call must be transferred to a human."
                            )
                        },
                        required=["reason"]
                    )
                ),
                # 4. log_call_summary
                types.FunctionDeclaration(
                    name="log_call_summary",
                    description="Store a structured post-call summary and follow-up action items in the permanent database.",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "summary": types.Schema(
                                type=types.Type.STRING,
                                description="Concise factual summary of the topics discussed and decisions made during the call."
                            ),
                            "action_items": types.Schema(
                                type=types.Type.ARRAY,
                                items=types.Schema(type=types.Type.STRING),
                                description="List of specific follow-up tasks or promises made to the caller."
                            )
                        },
                        required=["summary"]
                    )
                )
            ]
        )
    ]


async def _default_take_message(
    caller_name: str = "",
    phone_number: str = "",
    message: str = "",
    urgency: str = "medium",
    **kwargs
) -> Dict[str, Any]:
    """Default handler for take_message."""
    logger.info(f"Tool take_message invoked: caller={caller_name}, phone={phone_number}, urgency={urgency}")
    return {
        "status": "saved",
        "caller_name": caller_name,
        "phone_number": phone_number,
        "message": message,
        "urgency": urgency or "medium",
        "info": "Message saved successfully.",
    }


async def _default_end_call(
    reason: str = "conversation_finished",
    **kwargs
) -> Dict[str, Any]:
    """Default handler for end_call."""
    logger.info(f"Tool end_call invoked: reason={reason}")
    return {
        "status": "call_ended",
        "reason": reason,
    }


async def _default_transfer_to_human(
    reason: str = "caller_requested_human",
    **kwargs
) -> Dict[str, Any]:
    """Default handler for transfer_to_human."""
    logger.info(f"Tool transfer_to_human invoked: reason={reason}")
    return {
        "status": "transfer_initiated",
        "on_hold": True,
        "reason": reason,
    }


async def _default_log_call_summary(
    summary: str = "",
    action_items: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Default handler for log_call_summary."""
    items = action_items or []
    logger.info(f"Tool log_call_summary invoked: items_count={len(items)}")
    return {
        "status": "logged",
        "summary": summary,
        "action_item_count": len(items),
        "action_items": items,
    }


class ToolRegistry:
    """
    Registry for managing and executing tool callbacks during live call sessions.
    Provides default functional handlers for standard tools and allows registering
    custom handler overrides.
    """

    def __init__(self):
        self._handlers: Dict[str, Callable] = {}
        # Register sensible default handlers
        self.register_handler("take_message", _default_take_message)
        self.register_handler("end_call", _default_end_call)
        self.register_handler("transfer_to_human", _default_transfer_to_human)
        self.register_handler("log_call_summary", _default_log_call_summary)

    def register_handler(
        self,
        name: str,
        handler: Callable[..., Union[Dict[str, Any], Awaitable[Dict[str, Any]]]]
    ) -> None:
        """Registers or replaces a handler for the given tool name."""
        self._handlers[name] = handler
        logger.debug(f"Registered tool handler for: {name}")

    def get_handler(self, name: str) -> Optional[Callable]:
        """Retrieves registered handler for tool name."""
        return self._handlers.get(name)

    async def execute_tool(self, name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes a registered tool handler with the given arguments.
        Dispatches both synchronous and asynchronous callbacks safely.
        """
        handler = self._handlers.get(name)
        if not handler:
            logger.warning(f"Attempted to execute unregistered tool: '{name}'")
            return {"status": "error", "error": f"Tool '{name}' not found"}

        if args is None:
            clean_args = {}
        elif not isinstance(args, dict):
            logger.warning(f"Invalid arguments payload for tool '{name}': expected dict, got {type(args).__name__}")
            return {
                "status": "error",
                "error": f"Invalid arguments payload: expected dictionary, got {type(args).__name__}"
            }
        else:
            try:
                clean_args = dict(args)
            except Exception as e:
                logger.warning(f"Could not convert arguments to dict for tool '{name}': {e}")
                return {
                    "status": "error",
                    "error": f"Invalid arguments payload: expected dictionary, got {type(args).__name__}"
                }

        try:
            sig = inspect.signature(handler)
            has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

            if has_var_keyword:
                res = handler(**clean_args)
            else:
                accepted_keys = set(sig.parameters.keys())
                filtered_args = {k: v for k, v in clean_args.items() if k in accepted_keys}
                all_have_defaults = all(p.default != inspect.Parameter.empty for p in sig.parameters.values())
                if filtered_args or len(sig.parameters) == 0 or all_have_defaults:
                    res = handler(**filtered_args)
                else:
                    # If handler expects positional dict
                    res = handler(clean_args)

            if inspect.isawaitable(res):
                res = await res

            if isinstance(res, dict):
                return res
            return {"status": "ok", "result": res}

        except Exception as e:
            logger.error(f"Error executing tool '{name}': {e}", exc_info=True)
            return {"status": "error", "error": str(e)}
