# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""默认配置的参数化设置。"""

from typing_extensions import NotRequired

from .llm_config_input import LLMConfigInput


class EntityExtractionConfigInput(LLMConfigInput):
    """
    实体提取配置部分。

    该类定义了实体提取配置的参数，包括提示语、实体类型、最大获取数量、策略和编码模型。
    """

    # 提示语
    prompt: NotRequired[str | None]  # 可选，提示语

    # 实体类型
    entity_types: NotRequired[list[str] | str | None]  # 可选，实体类型

    # 最大获取数量
    max_gleanings: NotRequired[int | str | None]  # 可选，最大获取数量

    # 策略
    strategy: NotRequired[dict | None]  # 可选，策略

    # 编码模型
    encoding_model: NotRequired[str | None]  # 可选，编码模型
