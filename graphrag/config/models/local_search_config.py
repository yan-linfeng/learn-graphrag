# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""
配置模型的默认配置参数。
"""

from pydantic import BaseModel, Field

import graphrag.config.defaults as defs


class LocalSearchConfig(BaseModel):
    """
    本类定义了本地搜索配置的默认设置，包括文本单元比例、社区比例、对话历史最大回合数、映射实体的前K个、映射关系的前K个、温度、顶部概率、生成的完成次数、最大令牌数、LLM最大令牌数。
    """

    text_unit_prop: float = Field(
        description="文本单元比例。",
        default=defs.LOCAL_SEARCH_TEXT_UNIT_PROP,
    )
    community_prop: float = Field(
        description="社区比例。",
        default=defs.LOCAL_SEARCH_COMMUNITY_PROP,
    )
    conversation_history_max_turns: int = Field(
        description="对话历史最大回合数。",
        default=defs.LOCAL_SEARCH_CONVERSATION_HISTORY_MAX_TURNS,
    )
    top_k_entities: int = Field(
        description="映射实体的前K个。",
        default=defs.LOCAL_SEARCH_TOP_K_MAPPED_ENTITIES,
    )
    top_k_relationships: int = Field(
        description="映射关系的前K个。",
        default=defs.LOCAL_SEARCH_TOP_K_RELATIONSHIPS,
    )
    temperature: float | None = Field(
        description="生成令牌时使用的温度。",
        default=defs.LOCAL_SEARCH_LLM_TEMPERATURE,
    )
    top_p: float | None = Field(
        description="生成令牌时使用的顶部概率。",
        default=defs.LOCAL_SEARCH_LLM_TOP_P,
    )
    n: int | None = Field(
        description="生成的完成次数。",
        default=defs.LOCAL_SEARCH_LLM_N,
    )
    max_tokens: int = Field(
        description="最大令牌数。",
        default=defs.LOCAL_SEARCH_MAX_TOKENS,
    )
    llm_max_tokens: int = Field(
        description="LLM最大令牌数。",
        default=defs.LOCAL_SEARCH_LLM_MAX_TOKENS,
    )
