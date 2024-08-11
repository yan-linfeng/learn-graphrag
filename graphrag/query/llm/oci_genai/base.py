# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Base classes for LLM and Embedding models."""

from abc import ABC, abstractmethod
from typing import Dict, Any

import oci

from graphrag.query.llm.base import BaseTextEmbedding
from graphrag.query.llm.oci_genai.typing import OCIGenAIApiType
from graphrag.query.progress import ConsoleStatusReporter, StatusReporter


class BaseOCIGenAILLM(ABC):
    """The Base OCI Generative AI LLM implementation."""

    _client: oci.generative_ai_inference.GenerativeAiInferenceClient

    def __init__(self):
        self._create_oci_genai_client()

    @abstractmethod
    def _create_oci_genai_client(self):
        """Create a new OCI Generative AI client instance."""

    @property
    def client(self) -> oci.generative_ai_inference.GenerativeAiInferenceClient:
        """
        Get the client used for making API requests.

        Returns
        -------
            oci.generative_ai_inference.GenerativeAiInferenceClient: The client object.
        """
        return self._client

    @client.setter
    def client(self, client: oci.generative_ai_inference.GenerativeAiInferenceClient):
        """
        Set the client used for making API requests.

        Args:
            client (oci.generative_ai_inference.GenerativeAiInferenceClient): The client object.
        """
        self._client = client


class OCIGenAILLMImpl(BaseOCIGenAILLM):
    """OCI Generative AI LLM Implementation."""

    _reporter: StatusReporter = ConsoleStatusReporter()

    def __init__(
        self,
        config_file: str = '~/.oci/config',
        config_profile: str = "DEFAULT",
        compartment_id: str | None = None,
        endpoint: str = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        model_id: str | None = None,
        max_tokens: int = 4095,
        temperature: float = 1.0,
        top_p: float = 0.75,
        top_k: int = 0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        reporter: StatusReporter | None = None,
    ):
        self.config_profile = config_profile
        self.config_file = config_file
        self.compartment_id = compartment_id
        self.endpoint = endpoint
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.reporter = reporter or ConsoleStatusReporter()

        try:
            # Create OCI Generative AI client
            super().__init__()
        except Exception as e:
            self._reporter.error(
                message="Failed to create OCI Generative AI client",
                details={self.__class__.__name__: str(e)},
            )
            raise

    def _create_oci_genai_client(self):
        """Create a new OCI Generative AI client instance."""
        config = oci.config.from_file(self.config_file, self.config_profile)
        self._client = oci.generative_ai_inference.GenerativeAiInferenceClient(
            config=config,
            service_endpoint=self.endpoint,
            retry_strategy=oci.retry.NoneRetryStrategy(),
            timeout=(10, 240)
        )

    def chat(self, message: str) -> Dict[str, Any]:
        """
        Send a chat request to the OCI Generative AI service.

        Args:
            message (str): The input message for the chat.

        Returns:
            Dict[str, Any]: The response from the chat service.
        """
        chat_detail = oci.generative_ai_inference.models.ChatDetails()
        chat_request = oci.generative_ai_inference.models.CohereChatRequest()

        chat_request.message = message
        chat_request.max_tokens = self.max_tokens
        chat_request.temperature = self.temperature
        chat_request.top_p = self.top_p
        chat_request.top_k = self.top_k
        chat_request.frequency_penalty = self.frequency_penalty
        chat_request.presence_penalty = self.presence_penalty

        chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id=self.model_id)
        chat_detail.chat_request = chat_request
        chat_detail.compartment_id = self.compartment_id

        chat_response = self.client.chat(chat_detail)
        return vars(chat_response)


class OCIGenAITextEmbeddingImpl(BaseTextEmbedding):
    """OCI Generative AI Text Embedding Implementation."""

    _reporter: StatusReporter | None = None

    def _create_oci_genai_client(self, api_type: OCIGenAIApiType):
        """Create a new OCI Generative AI client instance."""
        # Implementation for text embedding would go here
        pass
