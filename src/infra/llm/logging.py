import logging
from typing import List, Optional
from .message import Message
from .action import Action
from .models import Models, ReasoningEffort
from src.infra.util.str_util import remove_indentation


class LLMLogger:
    """Centralized LLM request/response logging"""

    def __init__(self, logger=None, prompt_logger=None):
        """
        Args:
            logger: Standard logger for aggregated logs
            prompt_logger: Optional logger for per-message file logging (one file per request/response)
        """
        self.logger = logger.getChild("LLM") if logger else None
        self.prompt_logger = prompt_logger

    def log_request(
        self,
        model: Models,
        messages: List[Message],
        temperature: float,
        reasoning_effort: Optional[ReasoningEffort],
        tenant_id: Optional[int],
        session_id: Optional[str],
        log_tags: Optional[List[str]],
        available_actions: Optional[List[Action]],
        cache_path: Optional[str],
    ) -> None:
        """Log outgoing LLM request"""

        available_actions_str = ""
        if available_actions:
            available_actions_str = "## Available Actions\n"
            available_actions_str += "\n".join(
                [f"- {action.name}" for action in available_actions]
            )

        messages_formatted = "".join(
            [
                f"{message.role.value.capitalize()}: {message.content.replace(chr(10), chr(10) + '  ')}\n"
                for message in messages
            ]
        )

        tags_str = f"/ Tags: {', '.join(log_tags)}" if log_tags else ""

        request_content = remove_indentation(
            f""".
            >>> LLM Request >>> Session ID: {session_id} {tags_str}
            ## Messages
            {messages_formatted}

            ## Call Params
            Tenant ID: {tenant_id} / Model: {model.model_name} / Temperature: {temperature} / Reasoning Effort: {reasoning_effort.value if reasoning_effort else None}

            ## Cache
            Cache: {cache_path if cache_path else "No cache"}

            {available_actions_str}
            """
        )

        # Standard aggregated logging
        if self.logger:
            self.logger.debug(request_content)

        # Per-message file logging (one file per request)
        if self.prompt_logger:
            # Create a log record with session_id for filename generation
            record = logging.LogRecord(
                name=self.prompt_logger.name,
                level=logging.DEBUG,
                pathname="",
                lineno=0,
                msg=request_content,
                args=(),
                exc_info=None,
            )
            record.session_id = session_id
            self.prompt_logger.handle(record)

    def log_response(
        self,
        response,
        response_time: float,
        session_id: Optional[str],
    ) -> None:
        """Log incoming LLM response with token usage"""

        message = (
            response.choices[0].message
            if response.choices and len(response.choices) > 0
            else None
        )
        content = getattr(message, "content", None)

        # Tool calls
        tool_calls = getattr(message, "tool_calls", None)
        tool_calls_str = ""
        if tool_calls:
            for call in tool_calls:
                tool_calls_str += f"- {call.function.name}(id:{call.id}) - args: {call.function.arguments}\n"
        else:
            tool_calls_str = "- No tool calls\n"

        # Token usage
        tokens_usage = response.usage
        prompt_tokens = tokens_usage.prompt_tokens
        completion_tokens = tokens_usage.completion_tokens
        total_tokens = tokens_usage.total_tokens

        response_content = remove_indentation(
            f""".
            <<< LLM Response <<< Session ID: {session_id}
            Content
            --------
            {content if content else "[no response]"}
            --------
            ({response_time:.2f}s to respond)

            Tool Calls:
            {tool_calls_str}

            Tokens Usage:
            {prompt_tokens} prompt tokens + {completion_tokens} completion tokens = {total_tokens} total tokens
            """
        )

        # Standard aggregated logging
        if self.logger:
            self.logger.debug(response_content)

        if self.prompt_logger:
            # Create a log record with session_id for filename generation
            record = logging.LogRecord(
                name=self.prompt_logger.name,
                level=logging.DEBUG,
                pathname="",
                lineno=0,
                msg=response_content,
                args=(),
                exc_info=None,
            )
            record.session_id = session_id
            self.prompt_logger.handle(record)
