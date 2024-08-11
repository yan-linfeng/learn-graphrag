# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""
配置默认设置的参数化设置。

本模块定义了全局搜索配置的默认设置。
"""

from pydantic import BaseModel, Field

import graphrag.config.defaults as defs


class GlobalSearchConfig(BaseModel):
    """
    全局搜索配置的默认设置。

    本类定义了全局搜索配置的默认设置，包括温度、顶部概率、生成的完成次数、最大上下文大小、数据LLM最大令牌数、映射LLM最大令牌数、归约LLM最大令牌数和并发请求数。
    """

    # 温度，用于生成令牌
    temperature: float | None = Field(
        description="用于生成令牌的温度。",
        default=defs.GLOBAL_SEARCH_LLM_TEMPERATURE,
    )
    # 顶部概率，用于生成令牌
    top_p: float | None = Field(
        description="用于生成令牌的顶部概率。",
        default=defs.GLOBAL_SEARCH_LLM_TOP_P,
    )
    # 生成的完成次数
    n: int | None = Field(
        description="生成的完成次数。",
        default=defs.GLOBAL_SEARCH_LLM_N,
    )
    # 最大上下文大小（以令牌为单位）
    max_tokens: int = Field(
        description="最大上下文大小（以令牌为单位）。",
        default=defs.GLOBAL_SEARCH_MAX_TOKENS,
    )
    # 数据LLM最大令牌数
    data_max_tokens: int = Field(
        description="数据LLM最大令牌数。",
        default=defs.GLOBAL_SEARCH_DATA_MAX_TOKENS,
    )
    # 映射LLM最大令牌数
    map_max_tokens: int = Field(
        description="映射LLM最大令牌数。",
        default=defs.GLOBAL_SEARCH_MAP_MAX_TOKENS,
    )
    # 归约LLM最大令牌数
    reduce_max_tokens: int = Field(
        description="归约LLM最大令牌数。",
        default=defs.GLOBAL_SEARCH_REDUCE_MAX_TOKENS,
    )
    # 并发请求数
    concurrency: int = Field(
        description="并发请求数。",
        default=defs.GLOBAL_SEARCH_CONCURRENCY,
    )