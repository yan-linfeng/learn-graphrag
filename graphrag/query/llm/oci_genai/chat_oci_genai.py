# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Chat-based OCI Generative AI LLM implementation."""

from typing import Any, List

import oci
from tenacity import (
    AsyncRetrying,
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from graphrag.query.llm.base import BaseLLM, BaseLLMCallback
from graphrag.query.llm.oci_genai import OCIGenAILLMImpl
from graphrag.query.progress import StatusReporter

_MODEL_REQUIRED_MSG = "model_id is required"


class ChatOCIGenAI(BaseLLM, OCIGenAILLMImpl):
    """Wrapper for OCI Generative AI ChatCompletion models."""

    def __init__(
        self,
        config_file: str = '~/.oci/config',
        config_profile: str = "DEFAULT",
        compartment_id: str | None = None,
        endpoint: str = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        model_id: str = 'cohere.command-r-plus',
        max_tokens: int = 600,
        temperature: float = 1.0,
        top_p: float = 0.75,
        top_k: int = 0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        reporter: StatusReporter | None = None,
    ):
        OCIGenAILLMImpl.__init__(
            self=self,
            config_file=config_file,
            config_profile=config_profile,
            compartment_id=compartment_id,
            endpoint=endpoint,
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            reporter=reporter,
        )

        self._create_oci_genai_client()

    def _create_oci_genai_client(self):
        """Create a new OCI Generative AI client instance."""
        config = oci.config.from_file(self.config_file, self.config_profile)
        self._client = oci.generative_ai_inference.GenerativeAiInferenceClient(
            config=config,
            service_endpoint=self.endpoint,
            retry_strategy=oci.retry.NoneRetryStrategy(),
            timeout=(10, 240)
        )

    def generate(
        self,
        messages: str | List[Any],
        streaming: bool = True,
        callbacks: List[BaseLLMCallback] | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate text."""
        try:
            retryer = Retrying(
                stop=stop_after_attempt(10),
                wait=wait_exponential_jitter(max=10),
                reraise=True,
                retry=retry_if_exception_type(oci.exceptions.ServiceError),
            )
            for attempt in retryer:
                with attempt:
                    return self._generate(
                        messages=messages,
                        streaming=streaming,
                        callbacks=callbacks,
                        **kwargs,
                    )
        except RetryError as e:
            self._reporter.error(
                message="Error at generate()", details={self.__class__.__name__: str(e)}
            )
            return ""

    async def agenerate(
        self,
        messages: str | List[Any],
        streaming: bool = True,
        callbacks: List[BaseLLMCallback] | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate text asynchronously."""
        try:
            retryer = AsyncRetrying(
                stop=stop_after_attempt(10),
                wait=wait_exponential_jitter(max=10),
                reraise=True,
                retry=retry_if_exception_type(oci.exceptions.ServiceError),
            )
            async for attempt in retryer:
                with attempt:
                    return await self._agenerate(
                        messages=messages,
                        streaming=streaming,
                        callbacks=callbacks,
                        **kwargs,
                    )
        except RetryError as e:
            self._reporter.error(f"Error at agenerate(): {e}")
            return ""

    def _generate(
        self,
        messages: str | List[Any],
        streaming: bool = True,
        callbacks: List[BaseLLMCallback] | None = None,
        **kwargs: Any,
    ) -> str:
        if not self.model_id:
            raise ValueError(_MODEL_REQUIRED_MSG)

        print("use chat_oci_genai")

        chat_detail = oci.generative_ai_inference.models.ChatDetails()
        chat_request = oci.generative_ai_inference.models.CohereChatRequest()

        chat_request.message = messages if isinstance(messages, str) else messages[-1]['content']
        print(f"{chat_request.message=}")

        chat_request.max_tokens = self.max_tokens
        chat_request.temperature = self.temperature
        chat_request.frequency_penalty = self.frequency_penalty
        chat_request.presence_penalty = self.presence_penalty
        chat_request.top_p = self.top_p
        chat_request.top_k = self.top_k

        chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id=self.model_id)
        chat_detail.chat_request = chat_request
        chat_detail.compartment_id = self.compartment_id

        response = self._client.chat(chat_detail)

        full_response = response.data.chat_response

        if callbacks:
            for callback in callbacks:
                callback.on_llm_new_token(full_response)
                callback.on_llm_stop(usage=None)  # OCI doesn't provide usage info in the same way

        return full_response.text

    async def _agenerate(
        self,
        messages: str | List[Any],
        streaming: bool = True,
        callbacks: List[BaseLLMCallback] | None = None,
        **kwargs: Any,
    ) -> str:
        # OCI doesn't provide a native async client, so we'll use the sync version
        return self._generate(messages, streaming, callbacks, **kwargs)
