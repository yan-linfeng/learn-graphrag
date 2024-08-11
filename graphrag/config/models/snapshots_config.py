# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""
参数化设置：默认配置。

该模块定义了默认配置的参数化设置。

"""

from pydantic import BaseModel, Field

import graphrag.config.defaults as defs


class SnapshotsConfig(BaseModel):
    """
    快照配置部分。

    本类定义了快照的配置设置，包括是否保存GraphML、是否保存原始实体以及是否保存顶级节点。

    """

    graphml: bool = Field(
        description="是否保存GraphML的标志。",
        default=defs.SNAPSHOTS_GRAPHML,
    )
    raw_entities: bool = Field(
        description="是否保存原始实体的标志。",
        default=defs.SNAPSHOTS_RAW_ENTITIES,
    )
    top_level_nodes: bool = Field(
        description="是否保存顶级节点的标志。",
        default=defs.SNAPSHOTS_TOP_LEVEL_NODES,
    )
