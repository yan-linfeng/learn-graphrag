# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""默认配置的参数设置。"""

from typing_extensions import NotRequired, TypedDict


class ChunkingConfigInput(TypedDict):
    """分块的配置部分。"""

    # 分块大小
    size: NotRequired[int | str | None]
    # 重叠
    overlap: NotRequired[int | str | None]
    # 按列分组
    group_by_columns: NotRequired[list[str] | str | None]
    # 策略
    strategy: NotRequired[dict | None]
