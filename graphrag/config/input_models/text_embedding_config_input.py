# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""默认配置的参数化设置。

该模块包含了默认配置的参数化设置。

导入的模块:
- typing_extensions: 提供了扩展的类型注解功能。
- graphrag.config.enums: 包含了 TextEmbeddingTarget 枚举类型。
- .llm_config_input: 包含了 LLMConfigInput 类。

TextEmbeddingConfigInput 类用于配置文本嵌入。

Attributes:
    batch_size (NotRequired[int | str | None]): 批处理大小。
    batch_max_tokens (NotRequired[int | str | None]): 每批最大token数。
    target (NotRequired[TextEmbeddingTarget | str | None]): 目标。
    skip (NotRequired[list[str] | str | None]): 跳过的列表。
    vector_store (NotRequired[dict | None]): 向量存储。
    strategy (NotRequired[dict | None]): 策略。
"""

from typing_extensions import NotRequired

from graphrag.config.enums import (
    TextEmbeddingTarget,
)

from .llm_config_input import LLMConfigInput


class TextEmbeddingConfigInput(LLMConfigInput):
    """用于配置文本嵌入的配置节。

    Attributes:
        batch_size (NotRequired[int | str | None]): 批处理大小。
        batch_max_tokens (NotRequired[int | str | None]): 每批最大token数。
        target (NotRequired[TextEmbeddingTarget | str | None]): 目标。
        skip (NotRequired[list[str] | str | None]): 跳过的列表。
        vector_store (NotRequired[dict | None]): 向量存储。
        strategy (NotRequired[dict | None]): 策略。
    """

    batch_size: NotRequired[int | str | None]
    batch_max_tokens: NotRequired[int | str | None]
    target: NotRequired[TextEmbeddingTarget | str | None]
    skip: NotRequired[list[str] | str | None]
    vector_store: NotRequired[dict | None]
    strategy: NotRequired[dict | None]