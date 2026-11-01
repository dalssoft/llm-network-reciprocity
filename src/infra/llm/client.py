import time
import asyncio
import uuid
from typing import List, Optional, Union, Dict
from openai import OpenAI, AsyncOpenAI, NotGiven
from pydantic import BaseModel

from .models import Models, ReasoningEffort
from .message import Message, MessageRole, BlobAttachment
from .action import Action, ActionCall, ActionChoice
from .formatting import MessageFormatter, ToolFormatter
from .logging import LLMLogger
from src.infra.logger.logger import project_logger


class LLMResponse:
    """Container for LLM response data"""

    def __init__(self, content: str | BaseModel, actions: List[ActionCall]):
        self.content = content
        self.actions = actions


class LLMClient:
    """Core LLM API client - single responsibility: call OpenAI/LiteLLM API"""

    def __init__(self, logger=None, prompt_logger=None):
        """
        Initialize LLM client.

        The client automatically uses OpenAI-compatible environment variables:
        - OPENAI_API_BASE: Base URL (defaults to OpenAI's API)
        - OPENAI_API_KEY: API key (required)

        For LiteLLM proxy, set these before creating the client:
        - OPENAI_API_BASE=http://localhost:4000/v1
        - OPENAI_API_KEY=<your_litellm_master_key>

        Args:
            logger: Standard logger for aggregated logs
            prompt_logger: Optional logger for per-message file logging (one file per request/response)
        """
        self.client = OpenAI()
        self.async_client = AsyncOpenAI()
        self.logger = LLMLogger(logger or project_logger, prompt_logger=prompt_logger)
        self.message_formatter = MessageFormatter()
        self.tool_formatter = ToolFormatter()

    @staticmethod
    def build_messages(
        system_instruction: str,
        messages: List[str],
        attachments: List[BlobAttachment] = None,
    ) -> List[Message]:
        """Helper to build message list for backward compatibility"""
        messages = [
            Message(role=MessageRole.SYSTEM, content=system_instruction),
            *[Message(role=MessageRole.USER, content=m) for m in messages if m],
        ]
        if attachments:
            messages.extend(attachments)
        return messages

    def _prepare_request_params(
        self,
        messages: List[Message],
        available_actions: Optional[List[Action]],
        log_tags: Optional[List[str]],
        extra_headers: Optional[Dict[str, str]],
        session_id: Optional[str],
        extra_body: Optional[Dict],
        reasoning_effort: Optional[ReasoningEffort],
        openai_kwargs: Dict,
    ):
        """Prepare common request parameters for API calls."""
        messages_openai = self.message_formatter.to_openai_batch(messages)
        tools = (
            self.tool_formatter.to_openai_batch(available_actions)
            if available_actions
            else NotGiven()
        )

        request_headers = {}
        if log_tags:
            request_headers["x-litellm-tags"] = ", ".join(log_tags)
        if extra_headers:
            request_headers.update(extra_headers)
        extra_headers_final = request_headers if request_headers else None

        extra_body_dict = {}
        if session_id:
            extra_body_dict["litellm_session_id"] = session_id
        if extra_body:
            extra_body_dict.update(extra_body)
        extra_body_final = extra_body_dict if extra_body_dict else None

        reasoning_effort_value = (
            reasoning_effort.value if reasoning_effort else NotGiven()
        )

        openai_kwargs_clean = {
            k: v for k, v in openai_kwargs.items() if k != "extra_body"
        }

        return (
            messages_openai,
            tools,
            extra_headers_final,
            extra_body_final,
            reasoning_effort_value,
            openai_kwargs_clean,
        )

    def _extract_response_content(
        self,
        response,
        use_structured_output: bool,
        available_actions: Optional[List[Action]],
    ):
        """Extract content and actions from API response."""
        content = (
            response.choices[0].message.content
            if not use_structured_output
            else response.choices[0].message.parsed
        )

        tool_calls = response.choices[0].message.tool_calls or []
        actions = self.tool_formatter.from_openai_batch(
            tool_calls, available_actions or []
        )

        return content, actions

    async def _handle_structured_output_fallback_async(
        self,
        content: str,
        response_format: type[BaseModel],
        tenant_id: Optional[int],
        log_session_id: str,
        log_tags: Optional[List[str]],
        extra_headers: Optional[Dict[str, str]],
        extra_body: Optional[Dict],
        structured_output_fallback_model: Models,
        openai_kwargs: Dict,
    ):
        """Handle fallback when model doesn't support structured output (async)."""
        # Validate that fallback model supports structured output to prevent infinite recursion
        if not structured_output_fallback_model.supports_structured_output:
            raise ValueError(
                f"Fallback model {structured_output_fallback_model.model_name} does not support structured output. "
                "This would cause infinite recursion."
            )

        structure_messages = self.build_messages(
            system_instruction="You are a helpful assistant that extracts structured data from free text.",
            messages=[f"Extract structured data from:\n\n{content}"],
        )

        fallback_kwargs = {k: v for k, v in openai_kwargs.items() if k != "max_tokens"}

        # Pass the validated fallback model to recursive call
        # Validation above ensures it supports structured output, preventing infinite recursion
        # Use log_session_id to maintain session correlation across fallback calls
        return await self.generate_async(
            model=structured_output_fallback_model,
            messages=structure_messages,
            temperature=0.0,
            tenant_id=tenant_id,
            available_actions=None,
            response_format=response_format,
            action_choice=ActionChoice.NONE_OR_ANY,
            reasoning_effort=None,
            session_id=log_session_id,
            log_tags=log_tags,
            extra_headers=extra_headers,
            extra_body=extra_body,
            structured_output_fallback_model=structured_output_fallback_model,
            **fallback_kwargs,
        )

    def _handle_structured_output_fallback(
        self,
        content: str,
        response_format: type[BaseModel],
        tenant_id: Optional[int],
        log_session_id: str,
        log_tags: Optional[List[str]],
        extra_headers: Optional[Dict[str, str]],
        extra_body: Optional[Dict],
        structured_output_fallback_model: Models,
        openai_kwargs: Dict,
    ):
        """Handle fallback when model doesn't support structured output (sync)."""
        # Validate that fallback model supports structured output to prevent infinite recursion
        if not structured_output_fallback_model.supports_structured_output:
            raise ValueError(
                f"Fallback model {structured_output_fallback_model.model_name} does not support structured output. "
                "This would cause infinite recursion."
            )

        structure_messages = self.build_messages(
            system_instruction="You are a helpful assistant that extracts structured data from free text.",
            messages=[f"Extract structured data from:\n\n{content}"],
        )

        fallback_kwargs = {k: v for k, v in openai_kwargs.items() if k != "max_tokens"}

        # Pass the validated fallback model to recursive call
        # Validation above ensures it supports structured output, preventing infinite recursion
        # Use log_session_id to maintain session correlation across fallback calls
        return self.generate(
            model=structured_output_fallback_model,
            messages=structure_messages,
            temperature=0.0,
            tenant_id=tenant_id,
            available_actions=None,
            response_format=response_format,
            action_choice=ActionChoice.NONE_OR_ANY,
            reasoning_effort=None,
            session_id=log_session_id,
            log_tags=log_tags,
            extra_headers=extra_headers,
            extra_body=extra_body,
            structured_output_fallback_model=structured_output_fallback_model,
            **fallback_kwargs,
        )

    def generate(
        self,
        model: Models,
        messages: List[Message],
        temperature: float = 1.0,
        tenant_id: Optional[int] = None,
        available_actions: Optional[List[Action]] = None,
        response_format: Optional[type[BaseModel]] = None,
        action_choice: ActionChoice = ActionChoice.NONE_OR_ANY,
        reasoning_effort: Optional[ReasoningEffort] = None,
        session_id: Optional[str] = None,
        log_tags: Optional[List[str]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        extra_body: Optional[Dict] = None,
        structured_output_fallback_model: Optional[
            Models
        ] = Models.GEMINI_2_5_FLASH_LITE,
        **openai_kwargs,
    ) -> LLMResponse:
        """
        Generate LLM response synchronously with full OpenAI SDK parameter support

        Args:
            model: Model to use
            messages: List of messages
            temperature: Sampling temperature (0.0-2.0)
            tenant_id: Tenant ID for tracking
            available_actions: Optional tools/functions
            response_format: Optional Pydantic schema for structured output
            action_choice: Tool choice strategy
            reasoning_effort: Reasoning effort level (for o1/o3/DeepSeek-R1)
            session_id: LiteLLM session tracking. If None, a unique session ID will be generated
                       for logging purposes (to match request/response pairs)
            log_tags: Logging tags
            extra_headers: Optional headers to add to the request
            extra_body: Optional dict to merge into extra_body (e.g., {"cache": {"no-cache": True}})
            **openai_kwargs: All other OpenAI SDK parameters passed through

        Returns:
            LLMResponse with content and actions
        """

        log_session_id = session_id if session_id else str(uuid.uuid4())
        self.logger.log_request(
            model=model,
            messages=messages,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tenant_id=tenant_id,
            session_id=log_session_id,
            log_tags=log_tags,
            available_actions=available_actions,
            cache_path=None,
        )

        start_time = time.time()
        (
            messages_openai,
            tools,
            extra_headers_final,
            extra_body_final,
            reasoning_effort_value,
            openai_kwargs_clean,
        ) = self._prepare_request_params(
            messages,
            available_actions,
            log_tags,
            extra_headers,
            log_session_id,  # Pass generated session ID so LiteLLM receives it
            extra_body,
            reasoning_effort,
            openai_kwargs,
        )

        needs_structured_output = response_format is not None
        model_supports_structured = model.supports_structured_output
        use_structured_output = needs_structured_output and model_supports_structured

        # Make API call
        if not use_structured_output:
            response = self.client.chat.completions.create(
                user=str(tenant_id) if tenant_id else NotGiven(),
                model=model.model_name,
                messages=messages_openai,
                tools=tools,
                tool_choice=(
                    action_choice.value
                    if not isinstance(tools, NotGiven)
                    else NotGiven()
                ),
                temperature=temperature,
                reasoning_effort=reasoning_effort_value,
                extra_headers=extra_headers_final,
                extra_body=extra_body_final,
                **openai_kwargs_clean,
            )
        else:
            response = self.client.beta.chat.completions.parse(
                user=str(tenant_id) if tenant_id else NotGiven(),
                model=model.model_name,
                messages=messages_openai,
                tools=tools,
                tool_choice=(
                    action_choice.value
                    if not isinstance(tools, NotGiven)
                    else NotGiven()
                ),
                temperature=temperature,
                response_format=response_format,
                reasoning_effort=reasoning_effort_value,
                extra_headers=extra_headers_final,
                extra_body=extra_body_final,
                **openai_kwargs_clean,
            )

        end_time = time.time()
        self.logger.log_response(response, end_time - start_time, log_session_id)

        content, actions = self._extract_response_content(
            response, use_structured_output, available_actions
        )

        if needs_structured_output and not model_supports_structured:
            return self._handle_structured_output_fallback(
                content,
                response_format,
                tenant_id,
                log_session_id,
                log_tags,
                extra_headers,
                extra_body,
                structured_output_fallback_model,
                openai_kwargs,
            )

        return LLMResponse(content, actions=actions)

    async def generate_async(
        self,
        model: Models,
        messages: List[Message],
        temperature: float = 1.0,
        tenant_id: Optional[int] = None,
        available_actions: Optional[List[Action]] = None,
        response_format: Optional[type[BaseModel]] = None,
        action_choice: ActionChoice = ActionChoice.NONE_OR_ANY,
        reasoning_effort: Optional[ReasoningEffort] = None,
        session_id: Optional[str] = None,
        log_tags: Optional[List[str]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        extra_body: Optional[Dict] = None,
        structured_output_fallback_model: Optional[
            Models
        ] = Models.GEMINI_2_5_FLASH_LITE,
        **openai_kwargs,
    ) -> LLMResponse:
        """
        Generate LLM response asynchronously with full OpenAI SDK parameter support

        Args:
            model: Model to use
            messages: List of messages
            temperature: Sampling temperature (0.0-2.0)
            tenant_id: Tenant ID for tracking
            available_actions: Optional tools/functions
            response_format: Optional Pydantic schema for structured output
            action_choice: Tool choice strategy
            reasoning_effort: Reasoning effort level (for o1/o3/DeepSeek-R1)
            session_id: LiteLLM session tracking. If None, a unique session ID will be generated
                       for logging purposes (to match request/response pairs)
            log_tags: Logging tags
            extra_body: Optional dict to merge into extra_body (e.g., {"cache": {"no-cache": True}})
            structured_output_fallback_model: Model to use for structuring output when primary model doesn't support it
            **openai_kwargs: All other OpenAI SDK parameters passed through

        Returns:
            LLMResponse with content and actions
        """

        log_session_id = session_id if session_id else str(uuid.uuid4())
        self.logger.log_request(
            model=model,
            messages=messages,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tenant_id=tenant_id,
            session_id=log_session_id,
            log_tags=log_tags,
            available_actions=available_actions,
            cache_path=None,
        )

        start_time = time.time()
        (
            messages_openai,
            tools,
            extra_headers_final,
            extra_body_final,
            reasoning_effort_value,
            openai_kwargs_clean,
        ) = self._prepare_request_params(
            messages,
            available_actions,
            log_tags,
            extra_headers,
            log_session_id,  # Pass generated session ID so LiteLLM receives it
            extra_body,
            reasoning_effort,
            openai_kwargs,
        )

        needs_structured_output = response_format is not None
        model_supports_structured = model.supports_structured_output
        use_structured_output = needs_structured_output and model_supports_structured

        # Make async API call
        if not use_structured_output:
            response = await self.async_client.chat.completions.create(
                user=str(tenant_id) if tenant_id else NotGiven(),
                model=model.model_name,
                messages=messages_openai,
                tools=tools,
                tool_choice=(
                    action_choice.value
                    if not isinstance(tools, NotGiven)
                    else NotGiven()
                ),
                temperature=temperature,
                reasoning_effort=reasoning_effort_value,
                extra_headers=extra_headers_final,
                extra_body=extra_body_final,
                **openai_kwargs_clean,
            )
        else:
            response = await self.async_client.beta.chat.completions.parse(
                user=str(tenant_id) if tenant_id else NotGiven(),
                model=model.model_name,
                messages=messages_openai,
                tools=tools,
                tool_choice=(
                    action_choice.value
                    if not isinstance(tools, NotGiven)
                    else NotGiven()
                ),
                temperature=temperature,
                response_format=response_format,
                reasoning_effort=reasoning_effort_value,
                extra_headers=extra_headers_final,
                extra_body=extra_body_final,
                **openai_kwargs_clean,
            )

        end_time = time.time()
        self.logger.log_response(response, end_time - start_time, log_session_id)

        content, actions = self._extract_response_content(
            response, use_structured_output, available_actions
        )

        if needs_structured_output and not model_supports_structured:
            return await self._handle_structured_output_fallback_async(
                content,
                response_format,
                tenant_id,
                log_session_id,
                log_tags,
                extra_headers,
                extra_body,
                structured_output_fallback_model,
                openai_kwargs,
            )

        return LLMResponse(content, actions=actions)
