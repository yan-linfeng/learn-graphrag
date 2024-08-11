# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""默认配置的参数化设置。"""

from typing_extensions import NotRequired, TypedDict


class EmbedGraphConfigInput(TypedDict):
    """Node2Vec的默认配置部分。"""

    enabled: NotRequired[bool | str | None]  # 启用状态
    num_walks: NotRequired[int | str | None]  # 步行次数
    walk_length: NotRequired[int | str | None]  # 步行长度
    window_size: NotRequired[int | str | None]  # 窗口大小
    iterations: NotRequired[int | str | None]  # 迭代次数
    random_seed: NotRequired[int | str | None]  # 随机种子
    strategy: NotRequired[dict | None]  # 策略
