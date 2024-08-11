# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""
配置默认设置的参数化设置
"""

from pydantic import BaseModel, Field

import graphrag.config.defaults as defs


class ClusterGraphConfig(BaseModel):
    """
    聚类图的配置节
    """

    # 最大聚类大小的默认值
    max_cluster_size: int = Field(
        description="使用的最大聚类大小", default=defs.MAX_CLUSTER_SIZE
    )
    # 聚类策略的默认值
    strategy: dict | None = Field(
        description="使用的聚类策略", default=None
    )

    def resolved_strategy(self) -> dict:
        """
        获取解析后的聚类策略
        """
        # 导入聚类策略类型
        from graphrag.index.verbs.graph.clustering import GraphCommunityStrategyType

        # 如果没有指定策略，则使用默认的Leiden策略
        return self.strategy or {
            "type": GraphCommunityStrategyType.leiden,
            "max_cluster_size": self.max_cluster_size,
        }