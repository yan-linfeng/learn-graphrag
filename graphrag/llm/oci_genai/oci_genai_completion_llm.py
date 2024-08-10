# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A text-completion based LLM."""

import logging
from typing import Dict, Any

import oci
from typing_extensions import Unpack

from graphrag.llm.base import BaseLLM
from graphrag.llm.types import (
    CompletionInput,
    CompletionOutput,
    LLMInput,
)
from .oci_genai_configuration import OCIGenAIConfiguration
from .types import OCIGenAIClientTypes

log = logging.getLogger(__name__)


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


class OCIGenAICompletionLLM(BaseLLM[CompletionInput, CompletionOutput]):
    """A text-completion based LLM."""

    _client: OCIGenAIClientTypes
    _configuration: OCIGenAIConfiguration

    def __init__(self, client: OCIGenAIClientTypes, configuration: OCIGenAIConfiguration):
        self.client = client
        self.configuration = configuration

    # async def _execute_llm(
    #     self,
    #     input: CompletionInput,
    #     **kwargs: Unpack[LLMInput],
    # ) -> CompletionOutput | None:
    #     args = get_completion_llm_args(
    #         kwargs.get("model_parameters"), self.configuration
    #     )
    #     completion = self.client.completions.create(prompt=input, **args)
    #     return completion.choices[0].text

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
