# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""配置默认设置的参数化设置。"""

from pydantic import Field

import graphrag.config.defaults as defs
from graphrag.config.enums import TextEmbeddingTarget

from .llm_config import LLMConfig


class TextEmbeddingConfig(LLMConfig):
    """
    文本嵌入配置节。

    本类定义了文本嵌入的配置参数，包括批大小、批最大令牌数、目标、跳过的嵌入、向量存储配置和策略覆盖配置。
    """

    # 批大小
    batch_size: int = Field(
        description="使用的批大小。", default=defs.EMBEDDING_BATCH_SIZE
    )

    # 批最大令牌数
    batch_max_tokens: int = Field(
        description="使用的批最大令牌数。", default=defs.EMBEDDING_BATCH_MAX_TOKENS
    )

    # 目标
    target: TextEmbeddingTarget = Field(
        description="使用的目标。可以是'all'或'required'。", default=defs.EMBEDDING_TARGET
    )

    # 跳过的嵌入
    skip: list[str] = Field(description="跳过的嵌入。", default=[])

    # 向量存储配置
    vector_store: dict | None = Field(
        description="向量存储配置。", default=None
    )

    # 策略覆盖配置
    strategy: dict | None = Field(
        description="策略覆盖配置。", default=None
    )

    def resolved_strategy(self) -> dict:
        """
        获取解析后的文本嵌入策略。

        如果策略覆盖配置不为空，则返回策略覆盖配置；否则，返回默认策略配置。
        """
        from graphrag.index.verbs.text.embed import TextEmbedStrategyType

        # 如果策略覆盖配置不为空，则返回策略覆盖配置
        if self.strategy:
            return self.strategy

        # 否则，返回默认策略配置
        return {
            # "type": TextEmbedStrategyType.openai,
            "type": TextEmbedStrategyType.oci_genai,
            "llm": self.llm.model_dump(),
            **self.parallelization.model_dump(),
            "batch_size": self.batch_size,
            "batch_max_tokens": self.batch_max_tokens,
        }