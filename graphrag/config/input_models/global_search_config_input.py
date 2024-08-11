# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""默认配置的参数化设置。"""

from typing_extensions import NotRequired, TypedDict


class GlobalSearchConfigInput(TypedDict):
    """
    全局搜索配置的默认配置节。

    该类定义了全局搜索配置的参数，包括最大令牌数、数据最大令牌数、映射最大令牌数、归约最大令牌数和并发度。
    """

    # 最大令牌数
    max_tokens: NotRequired[int | str | None]  # 可选，类型为整数或字符串或None

    # 数据最大令牌数
    data_max_tokens: NotRequired[int | str | None]  # 可选，类型为整数或字符串或None

    # 映射最大令牌数
    map_max_tokens: NotRequired[int | str | None]  # 可选，类型为整数或字符串或None

    # 归约最大令牌数
    reduce_max_tokens: NotRequired[int | str | None]  # 可选，类型为整数或字符串或None

    # 并发度
    concurrency: NotRequired[int | str | None]  # 可选，类型为整数或字符串或None
