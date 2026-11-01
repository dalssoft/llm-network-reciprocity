from typing import List, Dict, Any
import json
from llm_easy_tools import get_tool_defs

from .message import Message, BlobAttachment
from .action import Action, ActionCall


class MessageFormatter:
    """Convert messages to OpenAI format"""

    @staticmethod
    def to_openai(message: Message | BlobAttachment) -> dict:
        """Convert Message or BlobAttachment to OpenAI format"""
        if isinstance(message, BlobAttachment):
            return {
                "role": message.role.value,
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{message.mime_type};base64,{message.to_base64()}"
                        },
                    },
                ],
            }
        return {"role": message.role.value, "content": message.content}

    @staticmethod
    def to_openai_batch(messages: List[Message | BlobAttachment]) -> List[dict]:
        """Convert multiple messages to OpenAI format"""
        return [MessageFormatter.to_openai(m) for m in messages]


class ToolFormatter:
    """Convert tools/actions to OpenAI format"""

    @staticmethod
    def to_openai(action: Action) -> dict:
        """Convert Action to OpenAI tool format"""
        return get_tool_defs([action.function], strict=True)[0]

    @staticmethod
    def to_openai_batch(actions: List[Action]) -> List[dict]:
        """Convert multiple actions to OpenAI format"""
        return [ToolFormatter.to_openai(a) for a in actions]

    @staticmethod
    def from_openai(tool_call, available_actions: List[Action]) -> ActionCall:
        """Convert OpenAI tool call response to ActionCall"""
        action = next(
            (a for a in available_actions if a.name == tool_call.function.name),
            None
        )
        if not action:
            raise ValueError(
                f"No matching action found for {tool_call.function.name}"
            )

        return ActionCall(
            id=tool_call.id,
            action=action,
            arguments=json.loads(tool_call.function.arguments),
        )

    @staticmethod
    def from_openai_batch(tool_calls, available_actions: List[Action]) -> List[ActionCall]:
        """Convert multiple OpenAI tool calls to ActionCall objects"""
        return [
            ToolFormatter.from_openai(tc, available_actions)
            for tc in (tool_calls or [])
        ]
