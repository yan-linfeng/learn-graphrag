# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""The Chat-based language model."""

import logging
from typing import Dict, Any

import oci
from oci.generative_ai_inference.models import ChatResult
from typing_extensions import Unpack

from graphrag_original.llm.base import BaseLLM
from graphrag_original.llm.types import (
    CompletionInput,
    CompletionOutput,
    LLMInput,
    LLMOutput,
)
from ._prompts import JSON_CHECK_PROMPT
from .oci_genai_configuration import OCIGenAIConfiguration
from .types import OCIGenAIClientTypes
from .utils import (
    try_parse_json_object,
)

log = logging.getLogger(__name__)

_MAX_GENERATION_RETRIES = 3
FAILED_TO_CREATE_JSON_ERROR = "Failed to generate valid JSON output"


def get_chat_request(input: str, model_parameters: Dict[str, Any],
                     configuration: OCIGenAIConfiguration) -> oci.generative_ai_inference.models.CohereChatRequest:
    chat_request = oci.generative_ai_inference.models.CohereChatRequest()
    chat_request.message = input
    chat_request.max_tokens = model_parameters.get("max_tokens", configuration.max_tokens)
    chat_request.temperature = model_parameters.get("temperature", configuration.temperature)
    chat_request.frequency_penalty = model_parameters.get("frequency_penalty", 0)
    chat_request.top_p = model_parameters.get("top_p", configuration.top_p)
    chat_request.top_k = model_parameters.get("top_k", 0)
    return chat_request


class OCIGenAIChatLLM(BaseLLM[CompletionInput, CompletionOutput]):
    """A Chat-based LLM."""

    _client: OCIGenAIClientTypes
    _configuration: OCIGenAIConfiguration

    def __init__(self, client: OCIGenAIClientTypes, configuration: OCIGenAIConfiguration):
        self.client = client
        self.configuration = configuration

    # async def _execute_llm(
    #     self, input: CompletionInput, **kwargs: Unpack[LLMInput]
    # ) -> CompletionOutput | None:
    #     args = get_completion_llm_args(
    #         kwargs.get("model_parameters"), self.configuration
    #     )
    #     history = kwargs.get("history") or []
    #     messages = [
    #         *history,
    #         {"role": "user", "content": input},
    #     ]
    #     completion = await self.client.chat.completions.create(
    #         messages=messages, **args
    #     )
    #     return completion.choices[0].message.content

    async def _execute_llm(
        self, input: CompletionInput, **kwargs: Unpack[LLMInput]
    ) -> CompletionOutput | None:
        model_parameters = kwargs.get("model_parameters", {})
        history = kwargs.get("history") or []

        # Prepare the chat request
        chat_request = get_chat_request(input, model_parameters, self.configuration)

        # Prepare the chat details
        chat_detail = oci.generative_ai_inference.models.ChatDetails()
        chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
            model_id=self.configuration.model_id)
        chat_detail.chat_request = chat_request
        chat_detail.compartment_id = self.configuration.compartment_id

        # If there's a history, we need to add it to the chat request
        if history:
            chat_request.conversation = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in history
            ]

        # Execute the chat request
        chat_response = self.client.chat(chat_detail)
        print(f"{chat_response=}")
        print(f"{chat_response.data=}")

        # Extract the content from the response
        if chat_response.data and chat_response.data.chat_response and chat_response.data.chat_response.text:
            return chat_response.data.chat_response.text

        return None

    async def _invoke_json(
        self,
        input: CompletionInput,
        **kwargs: Unpack[LLMInput],
    ) -> LLMOutput[CompletionOutput]:
        """Generate JSON output."""
        name = kwargs.get("name") or "unknown"
        is_response_valid = kwargs.get("is_response_valid") or (lambda _x: True)

        async def generate(
            attempt: int | None = None,
        ) -> LLMOutput[CompletionOutput]:
            call_name = name if attempt is None else f"{name}@{attempt}"
            return (
                await self._native_json(input, **{**kwargs, "name": call_name})
                if self.configuration.model_supports_json
                else await self._manual_json(input, **{**kwargs, "name": call_name})
            )

        def is_valid(x: dict | None) -> bool:
            return x is not None and is_response_valid(x)

        result = await generate()
        retry = 0
        while not is_valid(result.json) and retry < _MAX_GENERATION_RETRIES:
            result = await generate(retry)
            retry += 1

        if is_valid(result.json):
            return result
        raise RuntimeError(FAILED_TO_CREATE_JSON_ERROR)

    async def _native_json(
        self, input: CompletionInput, **kwargs: Unpack[LLMInput]
    ) -> LLMOutput[CompletionOutput]:
        """Generate JSON output using a model's native JSON-output support."""
        result = await self._invoke(
            input,
            **{
                **kwargs,
                "model_parameters": {
                    **(kwargs.get("model_parameters") or {}),
                    "response_format": {"type": "json_object"},
                },
            },
        )

        output, json_output = try_parse_json_object(result.output or "")

        return LLMOutput[CompletionOutput](
            output=output,
            json=json_output,
            history=result.history,
        )

    async def _manual_json(
        self, input: CompletionInput, **kwargs: Unpack[LLMInput]
    ) -> LLMOutput[CompletionOutput]:
        # Otherwise, clean up the output and try to parse it as json
        result = await self._invoke(input, **kwargs)
        history = result.history or []
        output, json_output = try_parse_json_object(result.output or "")
        if json_output:
            return LLMOutput[CompletionOutput](
                output=result.output, json=json_output, history=history
            )
        # if not return correct formatted json, retry
        log.warning("error parsing llm json, retrying")

        # If cleaned up json is unparsable, use the LLM to reformat it (may throw)
        result = await self._try_clean_json_with_llm(output, **kwargs)
        output, json_output = try_parse_json_object(result.output or "")

        return LLMOutput[CompletionOutput](
            output=output,
            json=json_output,
            history=history,
        )

    async def _try_clean_json_with_llm(
        self, output: str, **kwargs: Unpack[LLMInput]
    ) -> LLMOutput[CompletionOutput]:
        name = kwargs.get("name") or "unknown"
        return await self._invoke(
            JSON_CHECK_PROMPT,
            **{
                **kwargs,
                "variables": {"input_text": output},
                "name": f"fix_json@{name}",
            },
        )
