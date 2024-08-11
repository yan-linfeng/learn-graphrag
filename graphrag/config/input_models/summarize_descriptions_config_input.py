# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""
参数化设置的默认配置。
"""

from typing_extensions import NotRequired

from .llm_config_input import LLMConfigInput


class SummarizeDescriptionsConfigInput(LLMConfigInput):
    """
    描述摘要摘要的配置部分。

    Attributes:
        prompt (NotRequired[str | None]): 摘要的提示。
        max_length (NotRequired[int | str | None]): 摘要的最大长度。
        strategy (NotRequired[dict | None]): 摘要的策略。
    """

    prompt: NotRequired[str | None]
    max_length: NotRequired[int | str | None]
    strategy: NotRequired[dict | None]
