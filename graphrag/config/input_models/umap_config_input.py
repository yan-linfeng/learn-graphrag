# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""
默认配置的参数化设置。
"""

from typing_extensions import NotRequired, TypedDict


class UmapConfigInput(TypedDict):
    """
    UMAP 配置节。

    该类定义了 UMAP 配置的参数。
    """

    # UMAP 启用状态
    enabled: NotRequired[bool | str | None]  # 启用状态（可选）
