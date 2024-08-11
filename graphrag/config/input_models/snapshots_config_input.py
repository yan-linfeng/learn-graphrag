# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""
默认配置的参数化设置。

本模块定义了快照配置的参数化设置。
"""

from typing_extensions import NotRequired, TypedDict


class SnapshotsConfigInput(TypedDict):
    """
    快照配置部分。

    本类定义了快照配置的参数，包括 GraphML、原始实体和顶级节点。
    """

    # GraphML 配置项
    graphml: NotRequired[bool | str | None]  # 是否启用 GraphML

    # 原始实体配置项
    raw_entities: NotRequired[bool | str | None]  # 是否启用原始实体

    # 顶级节点配置项
    top_level_nodes: NotRequired[bool | str | None]  # 是否启用顶级节点
