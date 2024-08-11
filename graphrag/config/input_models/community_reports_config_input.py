# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""
社区报告的默认配置参数设置。

该类定义了社区报告的配置部分，包括提示语、最大长度、最大输入长度和策略。
"""

from typing_extensions import NotRequired

from .llm_config_input import LLMConfigInput


class CommunityReportsConfigInput(LLMConfigInput):
    """
    社区报告的配置部分。

    该类定义了社区报告的配置参数，包括提示语、最大长度、最大输入长度和策略。
    """

    prompt: NotRequired[str | None]  # 提示语
    max_length: NotRequired[int | str | None]  # 最大长度
    max_input_length: NotRequired[int | str | None]  # 最大输入长度
    strategy: NotRequired[dict | None]  # 策略
