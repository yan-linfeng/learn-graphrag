# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""参数化设置默认配置"""

from pydantic import BaseModel, Field

import graphrag.config.defaults as defs


class EmbedGraphConfig(BaseModel):
    """
    默认配置节Node2Vec

    此类用于配置Node2Vec算法的参数
    """

    # 是否启用Node2Vec
    enabled: bool = Field(
        description="是否启用Node2Vec",
        default=defs.NODE2VEC_ENABLED,
    )
    # Node2Vec随机游走次数
    num_walks: int = Field(
        description="Node2Vec随机游走次数", default=defs.NODE2VEC_NUM_WALKS
    )
    # Node2Vec随机游走长度
    walk_length: int = Field(
        description="Node2Vec随机游走长度", default=defs.NODE2VEC_WALK_LENGTH
    )
    # Node2Vec窗口大小
    window_size: int = Field(
        description="Node2Vec窗口大小", default=defs.NODE2VEC_WINDOW_SIZE
    )
    # Node2Vec迭代次数
    iterations: int = Field(
        description="Node2Vec迭代次数", default=defs.NODE2VEC_ITERATIONS
    )
    # Node2Vec随机种子
    random_seed: int = Field(
        description="Node2Vec随机种子", default=defs.NODE2VEC_RANDOM_SEED
    )
    # 图嵌入策略覆盖
    strategy: dict | None = Field(
        description="图嵌入策略覆盖", default=None
    )

    def resolved_strategy(self) -> dict:
        """
        获取解析后的Node2Vec策略

        如果策略覆盖存在，则返回覆盖策略，否则返回默认策略
        """
        from graphrag.index.verbs.graph.embed import EmbedGraphStrategyType

        # 如果策略覆盖存在，则返回覆盖策略
        if self.strategy:
            return self.strategy
        # 否则返回默认策略
        else:
            return {
                "type": EmbedGraphStrategyType.node2vec,
                "num_walks": self.num_walks,
                "walk_length": self.walk_length,
                "window_size": self.window_size,
                "iterations": self.iterations,
                "random_seed": self.iterations,
            }
