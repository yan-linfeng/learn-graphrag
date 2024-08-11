# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""OpenAI Embedding model implementation."""

import asyncio
from typing import Any

import numpy as np
import oci
import tiktoken
from tenacity import (
    AsyncRetrying,
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from graphrag.query.llm.base import BaseTextEmbedding
from graphrag.query.llm.oci_genai import OCIGenAILLMImpl
from graphrag.query.llm.text_utils import chunk_text
from graphrag.query.progress import StatusReporter


class OCIGenAIEmbedding(BaseTextEmbedding, OCIGenAILLMImpl):
    """Wrapper for OpenAI Embedding models."""

    def __init__(
        self,
        config_file: str = '~/.oci/config',
        config_profile: str = "DEFAULT",
        compartment_id: str | None = None,
        endpoint: str = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        model_id: str = 'cohere.embed-multilingual-v3.0',
        max_tokens: int = 600,
        temperature: float = 1.0,
        top_p: float = 0.75,
        top_k: int = 0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        reporter: StatusReporter | None = None,
        encoding_name: str = "cl100k_base",
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
        self.token_encoder = tiktoken.get_encoding(encoding_name)

    def embed(self, text: str, **kwargs: Any) -> list[float]:
        """
        Embed text using OpenAI Embedding's sync function.

        For text longer than max_tokens, chunk texts into max_tokens, embed each chunk, then combine using weighted average.
        Please refer to: https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
        """
        return self._embed_with_retry(text, **kwargs)[0]
        # token_chunks = chunk_text(
        #     text=text, token_encoder=self.token_encoder, max_tokens=self.max_tokens
        # )
        # chunk_embeddings = []
        # chunk_lens = []
        # for chunk in token_chunks:
        #     try:
        #         embedding, chunk_len = self._embed_with_retry(chunk, **kwargs)
        #         chunk_embeddings.append(embedding)
        #         chunk_lens.append(chunk_len)
        #     # TODO: catch a more specific exception
        #     except Exception as e:  # noqa BLE001
        #         self._reporter.error(
        #             message="Error embedding chunk",
        #             details={self.__class__.__name__: str(e)},
        #         )
        #
        #         continue
        # chunk_embeddings = np.average(chunk_embeddings, axis=0, weights=chunk_lens)
        # chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings)
        # return chunk_embeddings.tolist()

    async def aembed(self, text: str, **kwargs: Any) -> list[float]:
        """
        Embed text using OpenAI Embedding's async function.

        For text longer than max_tokens, chunk texts into max_tokens, embed each chunk, then combine using weighted average.
        """
        return self._embed_with_retry(text, **kwargs)[0]

        # token_chunks = chunk_text(
        #     text=text, token_encoder=self.token_encoder, max_tokens=self.max_tokens
        # )
        # chunk_embeddings = []
        # chunk_lens = []
        # embedding_results = await asyncio.gather(*[
        #     self._aembed_with_retry(chunk, **kwargs) for chunk in token_chunks
        # ])
        # embedding_results = [result for result in embedding_results if result[0]]
        # chunk_embeddings = [result[0] for result in embedding_results]
        # chunk_lens = [result[1] for result in embedding_results]
        # chunk_embeddings = np.average(chunk_embeddings, axis=0, weights=chunk_lens)  # type: ignore
        # chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings)
        # return chunk_embeddings.tolist()

    def _embed_with_retry(
        self, text: str | tuple, **kwargs: Any
    ) -> tuple[list[float], int]:
        try:
            retryer = Retrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential_jitter(max=10),
                reraise=True,
                retry=retry_if_exception_type(oci.exceptions.ServiceError),
            )
            for attempt in retryer:
                with attempt:
                    embed_text_detail = oci.generative_ai_inference.models.EmbedTextDetails()
                    embed_text_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
                        model_id=self.model_id)
                    # print(f"{text=}")
                    # print(f"{self.compartment_id=}")
                    # print(f"{self.model_id=}")
                    embed_text_detail.inputs = [text]
                    embed_text_detail.truncate = "NONE"
                    embed_text_detail.compartment_id = self.compartment_id
                    embed_text_response = self._client.embed_text(embed_text_detail)
                    # print(f"{embed_text_response.data=}")
                    embedding = (
                        [embedding for embedding in embed_text_response.data.embeddings][0]
                        or []
                    )
                    return (embedding, len(text))
        except RetryError as e:
            self._reporter.error(
                message="Error at embed_with_retry()",
                details={self.__class__.__name__: str(e)},
            )
            return ([], 0)
        else:
            # TODO: why not just throw in this case?
            return ([], 0)

    async def _aembed_with_retry(
        self, text: str | tuple, **kwargs: Any
    ) -> tuple[list[float], int]:
        try:
            retryer = AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential_jitter(max=10),
                reraise=True,
                retry=retry_if_exception_type(oci.exceptions.ServiceError),
            )
            async for attempt in retryer:
                with attempt:
                    embed_text_detail = oci.generative_ai_inference.models.EmbedTextDetails()
                    embed_text_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
                        model_id=self.model_id)
                    print(f"{text=}")
                    print(f"{self.compartment_id=}")
                    print(f"{self.model_id=}")
                    embed_text_detail.inputs = [text]
                    embed_text_detail.truncate = "NONE"
                    embed_text_detail.compartment_id = self.compartment_id
                    embed_text_response = self._client.embed_text(embed_text_detail)
                    # print(f"{embed_text_response.data=}")
                    embedding = (
                        [embedding for embedding in embed_text_response.data.embeddings][0]
                        or []
                    )
                    return (embedding, len(text))
        except RetryError as e:
            self._reporter.error(
                message="Error at embed_with_retry()",
                details={self.__class__.__name__: str(e)},
            )
            return ([], 0)
        else:
            # TODO: why not just throw in this case?
            return ([], 0)
