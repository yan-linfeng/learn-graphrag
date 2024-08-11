# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""配置参数化的默认设置。"""

from pydantic import BaseModel, Field

import graphrag.config.defaults as defs


class ChunkingConfig(BaseModel):
    """
    分块配置部分。

    该配置部分用于定义分块的相关参数，包括分块大小、重叠度、分组列等。
    """

    # 分块大小
    size: int = Field(description="要使用的分块大小。", default=defs.CHUNK_SIZE)
    # 重叠度
    overlap: int = Field(description="要使用的重叠度。", default=defs.CHUNK_OVERLAP)
    # 分组列
    group_by_columns: list[str] = Field(
        description="要使用的分组列。", default=defs.CHUNK_GROUP_BY_COLUMNS
    )
    # 分块策略
    strategy: dict | None = Field(
        description="要使用的分块策略，覆盖默认的标记化策略。",
        default=None,
    )
    # 编码模型
    encoding_model: str | None = Field(
        default=None, description="要使用的编码模型。"
    )

    def resolved_strategy(self, encoding_model: str) -> dict:
        """
        获取解析后的分块策略。

        如果未指定策略，则使用默认的标记化策略。

        :param encoding_model: 编码模型
        :return: 解析后的分块策略
        """
        from graphrag.index.verbs.text.chunk import ChunkStrategyType

        # 如果未指定策略，则使用默认的标记化策略
        return self.strategy or {
            "type": ChunkStrategyType.tokens,
            "chunk_size": self.size,
            "chunk_overlap": self.overlap,
            "group_by_columns": self.group_by_columns,
            "encoding_name": self.encoding_model or encoding_model,
        }
