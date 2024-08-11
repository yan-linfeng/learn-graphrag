# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""默认配置的参数化设置。"""

from typing_extensions import NotRequired

from .llm_config_input import LLMConfigInput


class ClaimExtractionConfigInput(LLMConfigInput):
    """
    声明提取配置部分。

    该类定义了声明提取配置的参数，包括启用状态、提示语、描述、最大获取数量、策略和编码模型。
    """

    # 启用状态，是否启用声明提取功能
    enabled: NotRequired[bool | None]  # 启用状态

    # 提示语，用于提示用户输入
    prompt: NotRequired[str | None]  # 提示语

    # 描述，用于描述声明提取的目的
    description: NotRequired[str | None]  # 描述

    # 最大获取数量，用于限制获取的声明数量
    max_gleanings: NotRequired[int | str | None]  # 最大获取数量

    # 策略，用于定义声明提取的策略
    strategy: NotRequired[dict | None]  # 策略

    # 编码模型，用于定义声明提取的编码模型
    encoding_model: NotRequired[str | None]  # 编码模型
