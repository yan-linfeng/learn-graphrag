# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""默认配置的参数设置。"""

# 导入必要的模块
from datashaper import AsyncType
from pydantic import BaseModel, Field

# 导入默认配置模块
import graphrag.config.defaults as defs

# 导入LLM参数和并行化参数模块
from .llm_parameters import LLMParameters
from .parallelization_parameters import ParallelizationParameters


class LLMConfig(BaseModel):
    """
    基类：LLM配置步骤。

    该类定义了LLM配置步骤的基本结构，包括LLM参数、并行化参数和异步模式。
    """

    # LLM参数
    llm: LLMParameters = Field(
        description="要使用的LLM配置。", default=LLMParameters()
    )
    # 并行化参数
    parallelization: ParallelizationParameters = Field(
        description="要使用的并行化配置。", default=ParallelizationParameters()
    )
    # 异步模式
    async_mode: AsyncType = Field(
        description="要使用的异步模式。", default=defs.ASYNC_MODE
    )
