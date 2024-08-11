# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""LLM 参数模型"""

from pydantic import BaseModel, Field

import graphrag.config.defaults as defs


class ParallelizationParameters(BaseModel):
    """
    并行化参数模型

    该模型定义了 LLM 服务的并行化参数，包括 stagger 和 num_threads。
    """

    # stagger 参数：LLM 服务的并行化 stagger 值
    stagger: float = Field(
        description="用于 LLM 服务的并行化 stagger 值",
        default=defs.PARALLELIZATION_STAGGER,
    )

    # num_threads 参数：LLM 服务的并行化线程数
    num_threads: int = Field(
        description="用于 LLM 服务的并行化线程数",
        default=defs.PARALLELIZATION_NUM_THREADS,
    )
