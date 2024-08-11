# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""
默认配置参数化设置的基类。

该类定义了默认配置的参数化设置，包括报告、存储、缓存、输入、嵌入图、嵌入向量、分块、快照、实体提取、摘要描述、社区报告、声明提取、聚类图、Umap、编码模型、跳过工作流程、本地搜索和全局搜索等配置。

注：该类继承自LLMConfigInput。
"""

from typing_extensions import NotRequired

from .cache_config_input import CacheConfigInput
from .chunking_config_input import ChunkingConfigInput
from .claim_extraction_config_input import ClaimExtractionConfigInput
from .cluster_graph_config_input import ClusterGraphConfigInput
from .community_reports_config_input import CommunityReportsConfigInput
from .embed_graph_config_input import EmbedGraphConfigInput
from .entity_extraction_config_input import EntityExtractionConfigInput
from .global_search_config_input import GlobalSearchConfigInput
from .input_config_input import InputConfigInput
from .llm_config_input import LLMConfigInput
from .local_search_config_input import LocalSearchConfigInput
from .reporting_config_input import ReportingConfigInput
from .snapshots_config_input import SnapshotsConfigInput
from .storage_config_input import StorageConfigInput
from .summarize_descriptions_config_input import (
    SummarizeDescriptionsConfigInput,
)
from .text_embedding_config_input import TextEmbeddingConfigInput
from .umap_config_input import UmapConfigInput


class GraphRagConfigInput(LLMConfigInput):
    """
    默认配置参数化设置的基类。

    该类定义了默认配置的参数化设置，包括报告、存储、缓存、输入、嵌入图、嵌入向量、分块、快照、实体提取、摘要描述、社区报告、声明提取、聚类图、Umap、编码模型、跳过工作流程、本地搜索和全局搜索等配置。

    注：该类继承自LLMConfigInput。
    """

    reporting: NotRequired[ReportingConfigInput | None]  # 报告配置
    storage: NotRequired[StorageConfigInput | None]  # 存储配置
    cache: NotRequired[CacheConfigInput | None]  # 缓存配置
    input: NotRequired[InputConfigInput | None]  # 输入配置
    embed_graph: NotRequired[EmbedGraphConfigInput | None]  # 嵌入图配置
    embeddings: NotRequired[TextEmbeddingConfigInput | None]  # 嵌入向量配置
    chunks: NotRequired[ChunkingConfigInput | None]  # 分块配置
    snapshots: NotRequired[SnapshotsConfigInput | None]  # 快照配置
    entity_extraction: NotRequired[EntityExtractionConfigInput | None]  # 实体提取配置
    summarize_descriptions: NotRequired[SummarizeDescriptionsConfigInput | None]  # 摘要描述配置
    community_reports: NotRequired[CommunityReportsConfigInput | None]  # 社区报告配置
    claim_extraction: NotRequired[ClaimExtractionConfigInput | None]  # 声明提取配置
    cluster_graph: NotRequired[ClusterGraphConfigInput | None]  # 聚类图配置
    umap: NotRequired[UmapConfigInput | None]  # Umap配置
    encoding_model: NotRequired[str | None]  # 编码模型
    skip_workflows: NotRequired[list[str] | str | None]  # 跳过工作流程
    local_search: NotRequired[LocalSearchConfigInput | None]  # 本地搜索配置
    global_search: NotRequired[GlobalSearchConfigInput | None]  # 全局搜索配置
