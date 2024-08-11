# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License


"""参数化设置，用于默认配置"""

from devtools import pformat
from pydantic import Field

import graphrag.config.defaults as defs
from .cache_config import CacheConfig
from .chunking_config import ChunkingConfig
from .claim_extraction_config import ClaimExtractionConfig
from .cluster_graph_config import ClusterGraphConfig
from .community_reports_config import CommunityReportsConfig
from .embed_graph_config import EmbedGraphConfig
from .entity_extraction_config import EntityExtractionConfig
from .global_search_config import GlobalSearchConfig
from .input_config import InputConfig
from .llm_config import LLMConfig
from .local_search_config import LocalSearchConfig
from .reporting_config import ReportingConfig
from .snapshots_config import SnapshotsConfig
from .storage_config import StorageConfig
from .summarize_descriptions_config import (
    SummarizeDescriptionsConfig,
)
from .text_embedding_config import TextEmbeddingConfig
from .umap_config import UmapConfig


class GraphRagConfig(LLMConfig):
    """
    默认配置参数化设置的基础类。
    """

    def __repr__(self) -> str:
        """
        获取字符串表示形式。
        """
        return pformat(self, highlight=False)

    def __str__(self):
        """
        获取字符串表示形式。
        """
        return self.model_dump_json(indent=4)

    root_dir: str = Field(
        description="配置的根目录", default=None
    )

    reporting: ReportingConfig = Field(
        description="报告配置", default=ReportingConfig()
    )
    """报告配置."""

    storage: StorageConfig = Field(
        description="存储配置", default=StorageConfig()
    )
    """存储配置."""

    cache: CacheConfig = Field(
        description="缓存配置", default=CacheConfig()
    )
    """缓存配置."""

    input: InputConfig = Field(
        description="输入配置", default=InputConfig()
    )
    """输入配置."""

    embed_graph: EmbedGraphConfig = Field(
        description="图嵌入配置", default=EmbedGraphConfig()
    )
    """图嵌入配置."""

    embeddings: TextEmbeddingConfig = Field(
        description="要使用的嵌入式LLM配置", default=TextEmbeddingConfig()
    )
    """要使用的嵌入式LLM配置."""

    chunks: ChunkingConfig = Field(
        description="分块配置", default=ChunkingConfig()
    )
    """分块配置."""

    snapshots: SnapshotsConfig = Field(
        description="快照配置", default=SnapshotsConfig()
    )
    """快照配置."""

    entity_extraction: EntityExtractionConfig = Field(
        description="实体提取配置", default=EntityExtractionConfig()
    )
    """实体提取配置."""

    summarize_descriptions: SummarizeDescriptionsConfig = Field(
        description="摘要描述配置", default=SummarizeDescriptionsConfig()
    )
    """摘要描述配置."""

    community_reports: CommunityReportsConfig = Field(
        description="社区报告配置", default=CommunityReportsConfig()
    )
    """社区报告配置."""

    claim_extraction: ClaimExtractionConfig = Field(
        description="声明提取配置", default=ClaimExtractionConfig(
            enabled=defs.CLAIM_EXTRACTION_ENABLED,
        ),
    )
    """声明提取配置."""

    cluster_graph: ClusterGraphConfig = Field(
        description="聚类图配置", default=ClusterGraphConfig()
    )
    """聚类图配置."""

    umap: UmapConfig = Field(
        description="UMAP配置", default=UmapConfig()
    )
    """UMAP配置."""

    local_search: LocalSearchConfig = Field(
        description="本地搜索配置", default=LocalSearchConfig()
    )
    """本地搜索配置."""

    global_search: GlobalSearchConfig = Field(
        description="全局搜索配置。",
        default=GlobalSearchConfig()
    )
    """全局搜索配置。"""

    encoding_model: str = Field(
        description="要使用的编码模型。",
        default=defs.ENCODING_MODEL
    )
    """要使用的编码模型。"""

    skip_workflows: list[str] = Field(
        description="要跳过的工作流程，通常用于测试目的。",
        default=[]
    )
    """要跳过的工作流程，通常用于测试目的。"""
