# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""默认配置的参数化设置。"""

# 导入必要的模块
from datashaper import AsyncType
from typing_extensions import NotRequired, TypedDict

# 导入相关的输入参数类
from .llm_parameters_input import LLMParametersInput
from .parallelization_parameters_input import ParallelizationParametersInput


class LLMConfigInput(TypedDict):
    """
    基础类：LLM 配置步骤的输入参数。

    该类定义了 LLM 配置步骤的输入参数，包括 LLM 参数、并行化参数和异步模式。
    """

    # LLM 参数（可选）
    llm: NotRequired[LLMParametersInput | None]  # LLM 参数输入

    # 并行化参数（可选）
    parallelization: NotRequired[ParallelizationParametersInput | None]  # 并行化参数输入

    # 异步模式（可选）
    async_mode: NotRequired[AsyncType | str | None]  # 异步模式输入
