# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""
参数化设置的默认配置。

本模块定义了本地搜索配置的参数化设置。
"""

from typing_extensions import NotRequired, TypedDict


class LocalSearchConfigInput(TypedDict):
    """
本地搜索配置的默认配置节。

本类定义了本地搜索配置的参数，包括文本单元属性、社区属性、对话历史最大回合数、实体数量、关系数量、最大令牌数和语言模型最大令牌数。
"""

    # 文本单元属性（可选）
    text_unit_prop: NotRequired[float | str | None]  # 可以是浮点数、字符串或None

    # 社区属性（可选）
    community_prop: NotRequired[float | str | None]  # 可以是浮点数、字符串或None

    # 对话历史最大回合数（可选）
    conversation_history_max_turns: NotRequired[int | str | None]  # 可以是整数、字符串或None

    # 实体数量（可选）
    top_k_entities: NotRequired[int | str | None]  # 可以是整数、字符串或None

    # 关系数量（可选）
    top_k_relationships: NotRequired[int | str | None]  # 可以是整数、字符串或None

    # 最大令牌数（可选）
    max_tokens: NotRequired[int | str | None]  # 可以是整数、字符串或None

    # 语言模型最大令牌数（可选）
    llm_max_tokens: NotRequired[int | str | None]  # 可以是整数、字符串或None
