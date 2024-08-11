# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""The EmbeddingsLLM class."""
from typing import List

import oci
from typing_extensions import Unpack

from graphrag_original.llm.base import BaseLLM
from graphrag_original.llm.types import (
    EmbeddingInput,
    EmbeddingOutput,
    LLMInput,
)

from .oci_genai_configuration import OCIGenAIConfiguration
from .types import OCIGenAIClientTypes


class OCIGenAIEmbeddingsLLM(BaseLLM[EmbeddingInput, EmbeddingOutput]):
    """A text-embedding generator LLM."""

    # _client: OCIGenAIClientTypes
    _client: oci.generative_ai_inference.GenerativeAiInferenceClient
    _configuration: OCIGenAIConfiguration

    # def __init__(self, client: OCIGenAIClientTypes, configuration: OCIGenAIConfiguration):
    #     self.client = client
    #     self.configuration = configuration
    def __init__(self, client: OCIGenAIClientTypes, configuration: OCIGenAIConfiguration):
        self.configuration = configuration
        config = oci.config.from_file('~/.oci/config', configuration.config_profile)
        self.client = oci.generative_ai_inference.GenerativeAiInferenceClient(
            config=config,
            service_endpoint=configuration.endpoint,
            retry_strategy=oci.retry.NoneRetryStrategy(),
            timeout=configuration.request_timeout
        )

    async def _execute_llm(
        self, input: EmbeddingInput, **kwargs: Unpack[LLMInput]
    ) -> EmbeddingOutput | None:
        embed_text_detail = oci.generative_ai_inference.models.EmbedTextDetails()
        embed_text_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
            model_id=self.configuration.model_id
        )
        embed_text_detail.inputs = input if isinstance(input, List) else [input]
        embed_text_detail.truncate = self.configuration.truncate
        embed_text_detail.compartment_id = self.configuration.compartment_id

        model_parameters = kwargs.get("model_parameters") or {}
        for key, value in model_parameters.items():
            setattr(embed_text_detail, key, value)

        embed_text_response = self.client.embed_text(embed_text_detail)
        print(f"{embed_text_response.data=}")

        # Assuming the response structure is similar to OpenAI's, adjust if necessary
        return [embedding for embedding in embed_text_response.data.embeddings]
