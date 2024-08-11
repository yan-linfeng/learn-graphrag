# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for the default configuration, loaded from environment variables."""

import os
from enum import Enum
from pathlib import Path
from typing import cast

from datashaper import AsyncType
from environs import Env
from pydantic import TypeAdapter

import graphrag.config.defaults as defs
from .enums import (
    CacheType,
    InputFileType,
    InputType,
    LLMType,
    ReportingType,
    StorageType,
    TextEmbeddingTarget,
)
from .environment_reader import EnvironmentReader
from .errors import (
    ApiKeyMissingError,
    AzureApiBaseMissingError,
    AzureDeploymentNameMissingError,
)
from .input_models import (
    GraphRagConfigInput,
    LLMConfigInput,
)
from .models import (
    CacheConfig,
    ChunkingConfig,
    ClaimExtractionConfig,
    ClusterGraphConfig,
    CommunityReportsConfig,
    EmbedGraphConfig,
    EntityExtractionConfig,
    GlobalSearchConfig,
    GraphRagConfig,
    InputConfig,
    LLMParameters,
    LocalSearchConfig,
    ParallelizationParameters,
    ReportingConfig,
    SnapshotsConfig,
    StorageConfig,
    SummarizeDescriptionsConfig,
    TextEmbeddingConfig,
    UmapConfig,
)
from .read_dotenv import read_dotenv

InputModelValidator = TypeAdapter(GraphRagConfigInput)


def create_graphrag_config(
    values: GraphRagConfigInput | None = None, root_dir: str | None = None
) -> GraphRagConfig:
    """
    从字典中加载配置参数。

    :param values: 配置参数字典
    :param root_dir: 根目录
    :return: GraphRagConfig 对象
    """
    # 如果 values 为空，则设置为默认值 {}
    values = values or {}
    # 如果 root_dir 为空，则设置为当前工作目录
    root_dir = root_dir or str(Path.cwd())

    # 创建环境对象
    env = _make_env(root_dir)
    # 替换 token
    _token_replace(cast(dict, values))
    # 验证输入模型
    InputModelValidator.validate_python(values, strict=True)

    # 创建环境读取器
    reader = EnvironmentReader(env)

    def hydrate_async_type(input: LLMConfigInput, base: AsyncType) -> AsyncType:
        """
        异步类型hydrate函数。

        :param input: 输入配置
        :param base: 基础异步类型
        :return: 异步类型对象
        """
        # 获取异步模式值
        value = input.get(Fragment.async_mode)
        # 如果值不为空，则返回异步类型对象，否则返回基础异步类型
        return AsyncType(value) if value else base

    def hydrate_llm_params(
        config: LLMConfigInput, base: LLMParameters
    ) -> LLMParameters:
        """
        LLM参数hydrate函数。

        :param config: 输入配置
        :param base: 基础LLM参数
        :return: LLM参数对象
        """
        # 使用环境读取器读取配置
        with reader.use(config.get("llm")):
            # 获取LLM类型
            llm_type = reader.str(Fragment.type)
            # 如果LLM类型不为空，则设置为LLM类型，否则设置为基础LLM类型
            llm_type = LLMType(llm_type) if llm_type else base.type
            # 获取API密钥
            api_key = reader.str(Fragment.api_key) or base.api_key
            # 获取API基础URL
            api_base = reader.str(Fragment.api_base) or base.api_base
            # 获取认知服务端点
            cognitive_services_endpoint = (
                reader.str(Fragment.cognitive_services_endpoint)
                or base.cognitive_services_endpoint
            )
            # 获取部署名称
            deployment_name = (
                reader.str(Fragment.deployment_name) or base.deployment_name
            )

            # 如果API密钥为空且LLM类型不是Azure，则抛出ApiKeyMissingError异常
            if api_key is None and not _is_azure(llm_type):
                raise ApiKeyMissingError
            # 如果LLM类型是Azure，则检查API基础URL和部署名称是否为空
            if _is_azure(llm_type):
                if api_base is None:
                    raise AzureApiBaseMissingError
                if deployment_name is None:
                    raise AzureDeploymentNameMissingError

            # 获取睡眠推荐值
            sleep_on_rate_limit = reader.bool(Fragment.sleep_recommendation)
            # 如果睡眠推荐值为空，则设置为基础睡眠推荐值
            if sleep_on_rate_limit is None:
                sleep_on_rate_limit = base.sleep_on_rate_limit_recommendation

            # 返回LLM参数对象
            return LLMParameters(
                api_key=api_key,
                type=llm_type,
                api_base=api_base,
                endpoint=endpoint,
                api_version=reader.str(Fragment.api_version) or base.api_version,
                organization=reader.str("organization") or base.organization,
                config_profile=reader.str("config_profile") or base.config_profile,
                compartment_id=reader.str("compartment_id") or base.compartment_id,
                proxy=reader.str("proxy") or base.proxy,
                model=reader.str("model") or base.model,
                model_id=reader.str("model_id") or base.model_id,
                max_tokens=reader.int(Fragment.max_tokens) or base.max_tokens,
                temperature=reader.float(Fragment.temperature) or base.temperature,
                top_p=reader.float(Fragment.top_p) or base.top_p,
                top_k=reader.int(Fragment.top_k) or base.top_k,
                n=reader.int(Fragment.n) or base.n,
                model_supports_json=reader.bool(Fragment.model_supports_json)
                                    or base.model_supports_json,
                request_timeout=reader.float(Fragment.request_timeout)
                                or base.request_timeout,
                cognitive_services_endpoint=cognitive_services_endpoint,
                deployment_name=deployment_name,
                tokens_per_minute=reader.int("tokens_per_minute", Fragment.tpm)
                                  or base.tokens_per_minute,
                requests_per_minute=reader.int("requests_per_minute", Fragment.rpm)
                                    or base.requests_per_minute,
                max_retries=reader.int(Fragment.max_retries) or base.max_retries,
                max_retry_wait=reader.float(Fragment.max_retry_wait)
                               or base.max_retry_wait,
                sleep_on_rate_limit_recommendation=sleep_on_rate_limit,
                concurrent_requests=reader.int(Fragment.concurrent_requests)
                                    or base.concurrent_requests,
            )

    def hydrate_embeddings_params(
        config: LLMConfigInput, base: LLMParameters
    ) -> LLMParameters:
        """
        从配置中读取嵌入参数并返回LLMParameters对象。

        :param config: 配置输入
        :param base: 基础LLM参数
        :return: LLMParameters对象
        """
        with reader.use(config.get("llm")):
            # 读取API类型
            api_type = reader.str(Fragment.type) or defs.EMBEDDING_TYPE
            api_type = LLMType(api_type) if api_type else defs.LLM_TYPE

            # 读取API密钥
            api_key = reader.str(Fragment.api_key) or base.api_key

            # 在某些特殊情况下，需要根据API类型来决定API基础URL
            # - 同一API基础URL用于LLM和嵌入（均为Azure）
            # - 不同API基础URL用于LLM和嵌入（均为Azure）
            # - LLM使用Azure OpenAI，而嵌入使用基础OpenAI（此情况重要）
            # - LLM使用Azure OpenAI，而嵌入使用第三方OpenAI-like API
            api_base = (
                reader.str(Fragment.api_base) or base.api_base
                if _is_azure(api_type)
                else reader.str(Fragment.api_base)
            )

            # 读取端点
            endpoint = (
                reader.str(Fragment.endpoint) or base.endpoint
                if _is_azure(api_type)
                else reader.str(Fragment.endpoint)
            )

            # 读取API版本
            api_version = (
                reader.str(Fragment.api_version) or base.api_version
                if _is_azure(api_type)
                else reader.str(Fragment.api_version)
            )

            # 读取组织
            api_organization = reader.str("organization") or base.organization

            # 读取配置文件
            config_profile = reader.str("config_profile") or base.config_profile

            # 读取隔间ID
            compartment_id = reader.str("compartment_id") or base.compartment_id

            # 读取代理
            api_proxy = reader.str("proxy") or base.proxy

            # 读取认知服务端点
            cognitive_services_endpoint = (
                reader.str(Fragment.cognitive_services_endpoint)
                or base.cognitive_services_endpoint
            )

            # 读取部署名称
            deployment_name = reader.str(Fragment.deployment_name)

            # 检查API密钥是否为空
            if api_key is None and not _is_azure(api_type):
                raise ApiKeyMissingError(embedding=True)

            # 检查Azure API基础URL和部署名称是否为空
            if _is_azure(api_type):
                if api_base is None:
                    raise AzureApiBaseMissingError(embedding=True)
                if deployment_name is None:
                    raise AzureDeploymentNameMissingError(embedding=True)

            # 读取睡眠推荐值
            sleep_on_rate_limit = reader.bool(Fragment.sleep_recommendation)
            if sleep_on_rate_limit is None:
                sleep_on_rate_limit = base.sleep_on_rate_limit_recommendation

            # 返回LLMParameters对象
            return LLMParameters(
                api_key=api_key,
                type=api_type,
                api_base=api_base,
                endpoint=endpoint,
                api_version=api_version,
                organization=api_organization,
                config_profile=config_profile,
                compartment_id=compartment_id,
                proxy=api_proxy,
                model=reader.str(Fragment.model) or defs.EMBEDDING_MODEL,
                model_id=reader.str(Fragment.model_id) or defs.EMBEDDING_MODEL_ID,
                request_timeout=reader.float(Fragment.request_timeout)
                                or defs.LLM_REQUEST_TIMEOUT,
                cognitive_services_endpoint=cognitive_services_endpoint,
                deployment_name=deployment_name,
                tokens_per_minute=reader.int("tokens_per_minute", Fragment.tpm)
                                  or defs.LLM_TOKENS_PER_MINUTE,
                requests_per_minute=reader.int("requests_per_minute", Fragment.rpm)
                                    or defs.LLM_REQUESTS_PER_MINUTE,
                max_retries=reader.int(Fragment.max_retries) or defs.LLM_MAX_RETRIES,
                max_retry_wait=reader.float(Fragment.max_retry_wait)
                               or defs.LLM_MAX_RETRY_WAIT,
                sleep_on_rate_limit_recommendation=sleep_on_rate_limit,
                concurrent_requests=reader.int(Fragment.concurrent_requests)
                                    or defs.LLM_CONCURRENT_REQUESTS,
            )

    def hydrate_parallelization_params(
        config: LLMConfigInput, base: ParallelizationParameters
    ) -> ParallelizationParameters:
        """
        并行化参数hydrate函数。

        从配置中读取并行化参数并返回ParallelizationParameters对象。

        :param config: 配置输入
        :param base: 基础并行化参数
        :return: ParallelizationParameters对象
        """
        # 使用环境读取器读取配置
        with reader.use(config.get("parallelization")):
            # 返回并行化参数对象
            return ParallelizationParameters(
                # 读取线程数
                num_threads=reader.int("num_threads", Fragment.thread_count)
                            or base.num_threads,
                # 读取线程延迟
                stagger=reader.float("stagger", Fragment.thread_stagger)
                        or base.stagger,
            )

    # 设置OpenAI API密钥的默认值
    fallback_oai_key = env("OPENAI_API_KEY", env("AZURE_OPENAI_API_KEY", None))
    # 设置OpenAI组织ID的默认值
    fallback_oai_org = env("OPENAI_ORG_ID", None)
    # 设置OpenAI基础URL的默认值
    fallback_oai_base = env("OPENAI_BASE_URL", None)
    # 设置OpenAI API版本的默认值
    fallback_oai_version = env("OPENAI_API_VERSION", None)

    # 使用环境变量前缀和值读取配置
    with reader.envvar_prefix(Section.graphrag), reader.use(values):
        # 读取异步模式
        async_mode = reader.str(Fragment.async_mode)
        # 如果异步模式不为空，则设置为AsyncType，否则设置为默认值
        async_mode = AsyncType(async_mode) if async_mode else defs.ASYNC_MODE

        # 读取OpenAI API密钥
        fallback_oai_key = reader.str(Fragment.api_key) or fallback_oai_key
        # 读取OpenAI组织ID
        fallback_oai_org = reader.str(Fragment.api_organization) or fallback_oai_org
        # 读取OpenAI基础URL
        fallback_oai_base = reader.str(Fragment.api_base) or fallback_oai_base
        # 读取OpenAI API版本
        fallback_oai_version = reader.str(Fragment.api_version) or fallback_oai_version
        # 读取OpenAI代理
        fallback_oai_proxy = reader.str(Fragment.api_proxy)

        # 读取LLM配置
        with reader.envvar_prefix(Section.llm):
            with reader.use(values.get("llm")):
                # 读取LLM类型
                llm_type = reader.str(Fragment.type)
                llm_type = LLMType(llm_type) if llm_type else defs.LLM_TYPE
                """
                LLM类型，取自配置文件或默认值
                :type llm_type: LLMType
                """

                # 读取API密钥
                api_key = reader.str(Fragment.api_key) or fallback_oai_key
                """
                API密钥，取自配置文件或环境变量
                :type api_key: str
                """

                # 读取组织ID
                api_organization = (
                    reader.str(Fragment.api_organization) or fallback_oai_org
                )
                """
                组织ID，取自配置文件或环境变量
                :type api_organization: str
                """

                # 读取配置文件
                config_profile = (
                    reader.str(Fragment.config_profile) or "DEFAULT"
                )
                """
                配置文件，取自配置文件或默认值
                :type config_profile: str
                """

                # 读取隔间ID
                compartment_id = (
                    reader.str(Fragment.compartment_id) or None
                )
                """
                隔间ID，取自配置文件或默认值
                :type compartment_id: str
                """

                # 读取API基础URL
                api_base = reader.str(Fragment.api_base) or fallback_oai_base
                """
                API基础URL，取自配置文件或环境变量
                :type api_base: str
                """

                # 读取端点
                endpoint = reader.str(Fragment.endpoint) or None
                """
                端点，取自配置文件或默认值
                :type endpoint: str
                """

                # 读取API版本
                api_version = reader.str(Fragment.api_version) or fallback_oai_version
                """
                API版本，取自配置文件或环境变量
                :type api_version: str
                """

                # 读取API代理
                api_proxy = reader.str(Fragment.api_proxy) or fallback_oai_proxy
                """
                API代理，取自配置文件或环境变量
                :type api_proxy: str
                """

                # 读取认知服务端点
                cognitive_services_endpoint = reader.str(
                    Fragment.cognitive_services_endpoint
                )
                """
                认知服务端点，取自配置文件
                :type cognitive_services_endpoint: str
                """

                # 读取部署名称
                deployment_name = reader.str(Fragment.deployment_name)
                """
                部署名称，取自配置文件
                :type deployment_name: str
                """

                # 检查API密钥和Azure API基础URL
                if api_key is None and not _is_azure(llm_type):
                    raise ApiKeyMissingError
                if _is_azure(llm_type):
                    if api_base is None:
                        raise AzureApiBaseMissingError
                    if deployment_name is None:
                        raise AzureDeploymentNameMissingError

                # 读取睡眠推荐值
                sleep_on_rate_limit = reader.bool(Fragment.sleep_recommendation)
                if sleep_on_rate_limit is None:
                    sleep_on_rate_limit = defs.LLM_SLEEP_ON_RATE_LIMIT_RECOMMENDATION
                """
                睡眠推荐值，取自配置文件或默认值
                :type sleep_on_rate_limit: bool
                """

                # 创建LLM参数对象
                llm_model = LLMParameters(
                    api_key=api_key,
                    api_base=api_base,
                    endpoint=endpoint,
                    api_version=api_version,
                    organization=api_organization,
                    config_profile=config_profile,
                    compartment_id=compartment_id,
                    proxy=api_proxy,
                    type=llm_type,
                    model=reader.str(Fragment.model) or defs.LLM_MODEL,
                    model_id=reader.str(Fragment.model_id) or defs.LLM_MODEL_ID,
                    max_tokens=reader.int(Fragment.max_tokens) or defs.LLM_MAX_TOKENS,
                    temperature=reader.float(Fragment.temperature)
                                or defs.LLM_TEMPERATURE,
                    top_p=reader.float(Fragment.top_p) or defs.LLM_TOP_P,
                    top_k=reader.float(Fragment.top_k) or defs.LLM_TOP_K,
                    n=reader.int(Fragment.n) or defs.LLM_N,
                    model_supports_json=reader.bool(Fragment.model_supports_json),
                    request_timeout=reader.float(Fragment.request_timeout)
                                    or defs.LLM_REQUEST_TIMEOUT,
                    cognitive_services_endpoint=cognitive_services_endpoint,
                    deployment_name=deployment_name,
                    tokens_per_minute=reader.int(Fragment.tpm)
                                      or defs.LLM_TOKENS_PER_MINUTE,
                    requests_per_minute=reader.int(Fragment.rpm)
                                        or defs.LLM_REQUESTS_PER_MINUTE,
                    max_retries=reader.int(Fragment.max_retries)
                                or defs.LLM_MAX_RETRIES,
                    max_retry_wait=reader.float(Fragment.max_retry_wait)
                                   or defs.LLM_MAX_RETRY_WAIT,
                    sleep_on_rate_limit_recommendation=sleep_on_rate_limit,
                    concurrent_requests=reader.int(Fragment.concurrent_requests)
                                        or defs.LLM_CONCURRENT_REQUESTS,
                )
            # 并行化参数配置
            with reader.use(values.get("parallelization")):
                # 创建并行化参数模型
                llm_parallelization_model = ParallelizationParameters(
                    # 并行化延迟（stagger）
                    stagger=reader.float("stagger", Fragment.thread_stagger)
                            or defs.PARALLELIZATION_STAGGER,
                    # 并行化线程数（num_threads）
                    num_threads=reader.int("num_threads", Fragment.thread_count)
                                or defs.PARALLELIZATION_NUM_THREADS,
                )

            # 嵌入配置
            embeddings_config = values.get("embeddings") or {}
            with reader.envvar_prefix(Section.embedding), reader.use(embeddings_config):
                # 嵌入目标（target）
                embeddings_target = reader.str("target")

                # 创建嵌入配置模型
                embeddings_model = TextEmbeddingConfig(
                    # 语言模型（llm）
                    llm=hydrate_embeddings_params(embeddings_config, llm_model),
                    # 并行化参数
                    parallelization=hydrate_parallelization_params(
                        embeddings_config, llm_parallelization_model
                    ),
                    # 向量存储（vector_store）
                    vector_store=embeddings_config.get("vector_store", None),
                    # 异步模式（async_mode）
                    async_mode=hydrate_async_type(embeddings_config, async_mode),
                    # 嵌入目标（target）
                    target=(
                        TextEmbeddingTarget(embeddings_target)
                        if embeddings_target
                        else defs.EMBEDDING_TARGET
                    ),
                    # 批量大小（batch_size）
                    batch_size=reader.int("batch_size") or defs.EMBEDDING_BATCH_SIZE,
                    # 批量最大token数（batch_max_tokens）
                    batch_max_tokens=reader.int("batch_max_tokens")
                                     or defs.EMBEDDING_BATCH_MAX_TOKENS,
                    # 跳过列表（skip）
                    skip=reader.list("skip") or [],
                )

            # Node2Vec图嵌入配置
            with (
                reader.envvar_prefix(Section.node2vec),
                reader.use(values.get("embed_graph")),
            ):
                # 创建Node2Vec图嵌入配置模型
                embed_graph_model = EmbedGraphConfig(
                    # 启用Node2Vec（enabled）
                    enabled=reader.bool(Fragment.enabled) or defs.NODE2VEC_ENABLED,
                    # 随机游走次数（num_walks）
                    num_walks=reader.int("num_walks") or defs.NODE2VEC_NUM_WALKS,
                    # 游走长度（walk_length）
                    walk_length=reader.int("walk_length") or defs.NODE2VEC_WALK_LENGTH,
                    # 窗口大小（window_size）
                    window_size=reader.int("window_size") or defs.NODE2VEC_WINDOW_SIZE,
                    # 迭代次数（iterations）
                    iterations=reader.int("iterations") or defs.NODE2VEC_ITERATIONS,
                    # 随机种子（random_seed）
                    random_seed=reader.int("random_seed") or defs.NODE2VEC_RANDOM_SEED,
                )
        # 输入配置读取
        with reader.envvar_prefix(Section.input), reader.use(values.get("input")):
            # 读取输入类型
            input_type = reader.str("type")
            # 读取文件类型
            file_type = reader.str(Fragment.file_type)

            # 输入配置模型
            input_model = InputConfig(
                # 文件类型
                file_type=(
                    InputFileType(file_type) if file_type else defs.INPUT_FILE_TYPE
                ),
                # 输入类型
                type=(InputType(input_type) if input_type else defs.INPUT_TYPE),
                # 编码
                encoding=reader.str("file_encoding", Fragment.encoding)
                         or defs.INPUT_FILE_ENCODING,
                # 基础目录
                base_dir=reader.str(Fragment.base_dir) or defs.INPUT_BASE_DIR,
                # 文件模式
                file_pattern=reader.str("file_pattern")
                             or (
                                 defs.INPUT_TEXT_PATTERN
                                 if file_type == InputFileType.text
                                 else defs.INPUT_CSV_PATTERN
                             ),
                # 源列
                source_column=reader.str("source_column"),
                # 时间戳列
                timestamp_column=reader.str("timestamp_column"),
                # 时间戳格式
                timestamp_format=reader.str("timestamp_format"),
                # 文本列
                text_column=reader.str("text_column") or defs.INPUT_TEXT_COLUMN,
                # 标题列
                title_column=reader.str("title_column"),
                # 文档属性列
                document_attribute_columns=reader.list("document_attribute_columns")
                                           or [],
                # 连接字符串
                connection_string=reader.str(Fragment.conn_string),
                # 存储账户Blob URL
                storage_account_blob_url=reader.str(Fragment.storage_account_blob_url),
                # 容器名称
                container_name=reader.str(Fragment.container_name),
            )

        # 缓存配置读取
        with reader.envvar_prefix(Section.cache), reader.use(values.get("cache")):
            # 读取缓存类型
            c_type = reader.str(Fragment.type)

            # 缓存配置模型
            cache_model = CacheConfig(
                # 缓存类型
                type=CacheType(c_type) if c_type else defs.CACHE_TYPE,
                # 连接字符串
                connection_string=reader.str(Fragment.conn_string),
                # 存储账户Blob URL
                storage_account_blob_url=reader.str(Fragment.storage_account_blob_url),
                # 容器名称
                container_name=reader.str(Fragment.container_name),
                # 基础目录
                base_dir=reader.str(Fragment.base_dir) or defs.CACHE_BASE_DIR,
            )
        # reporting 配置读取
        with (
            reader.envvar_prefix(Section.reporting),  # 设置环境变量前缀
            reader.use(values.get("reporting")),  # 使用 reporting 配置
        ):
            # 读取 reporting 类型
            r_type = reader.str(Fragment.type)
            reporting_model = ReportingConfig(
                # reporting 类型
                type=ReportingType(r_type) if r_type else defs.REPORTING_TYPE,
                # 连接字符串
                connection_string=reader.str(Fragment.conn_string),
                # 存储账户 Blob URL
                storage_account_blob_url=reader.str(Fragment.storage_account_blob_url),
                # 容器名称
                container_name=reader.str(Fragment.container_name),
                # 基础目录
                base_dir=reader.str(Fragment.base_dir) or defs.REPORTING_BASE_DIR,
            )

        # storage 配置读取
        with reader.envvar_prefix(Section.storage), reader.use(values.get("storage")):
            # 读取 storage 类型
            s_type = reader.str(Fragment.type)
            storage_model = StorageConfig(
                # storage 类型
                type=StorageType(s_type) if s_type else defs.STORAGE_TYPE,
                # 连接字符串
                connection_string=reader.str(Fragment.conn_string),
                # 存储账户 Blob URL
                storage_account_blob_url=reader.str(Fragment.storage_account_blob_url),
                # 容器名称
                container_name=reader.str(Fragment.container_name),
                # 基础目录
                base_dir=reader.str(Fragment.base_dir) or defs.STORAGE_BASE_DIR,
            )

        # chunk 配置读取
        with reader.envvar_prefix(Section.chunk), reader.use(values.get("chunks")):
            # 读取 group_by_columns 列表
            group_by_columns = reader.list("group_by_columns", "BY_COLUMNS")
            if group_by_columns is None:
                # 如果 group_by_columns 为空，则使用默认值
                group_by_columns = defs.CHUNK_GROUP_BY_COLUMNS

            chunks_model = ChunkingConfig(
                # chunk 大小
                size=reader.int("size") or defs.CHUNK_SIZE,
                # chunk 重叠大小
                overlap=reader.int("overlap") or defs.CHUNK_OVERLAP,
                # group_by_columns 列表
                group_by_columns=group_by_columns,
                # 编码模型
                encoding_model=reader.str(Fragment.encoding_model),
            )
        # 快照配置
        with (
            reader.envvar_prefix(Section.snapshot),  # 设置环境变量前缀
            reader.use(values.get("snapshots")),  # 使用快照配置
        ):
            # 创建快照模型
            snapshots_model = SnapshotsConfig(
                # 是否保存GraphML
                graphml=reader.bool("graphml") or defs.SNAPSHOTS_GRAPHML,
                # 是否保存原始实体
                raw_entities=reader.bool("raw_entities") or defs.SNAPSHOTS_RAW_ENTITIES,
                # 是否保存顶级节点
                top_level_nodes=reader.bool("top_level_nodes") or defs.SNAPSHOTS_TOP_LEVEL_NODES,
            )

        # UMAP配置
        with reader.envvar_prefix(Section.umap), reader.use(values.get("umap")):
            # 创建UMAP模型
            umap_model = UmapConfig(
                # 是否启用UMAP
                enabled=reader.bool(Fragment.enabled) or defs.UMAP_ENABLED,
            )

        # 实体提取配置
        entity_extraction_config = values.get("entity_extraction") or {}
        with (
            reader.envvar_prefix(Section.entity_extraction),  # 设置环境变量前缀
            reader.use(entity_extraction_config),  # 使用实体提取配置
        ):
            # 最大提取数量
            max_gleanings = reader.int(Fragment.max_gleanings)
            max_gleanings = (
                max_gleanings
                if max_gleanings is not None
                else defs.ENTITY_EXTRACTION_MAX_GLEANINGS
            )

            # 创建实体提取模型
            entity_extraction_model = EntityExtractionConfig(
                # 语言模型
                llm=hydrate_llm_params(entity_extraction_config, llm_model),
                # 并行化参数
                parallelization=hydrate_parallelization_params(
                    entity_extraction_config, llm_parallelization_model
                ),
                # 异步模式
                async_mode=hydrate_async_type(entity_extraction_config, async_mode),
                # 实体类型
                entity_types=reader.list("entity_types") or defs.ENTITY_EXTRACTION_ENTITY_TYPES,
                # 最大提取数量
                max_gleanings=max_gleanings,
                # 提示语
                prompt=reader.str("prompt", Fragment.prompt_file),
                # 编码模型
                encoding_model=reader.str(Fragment.encoding_model),
                # 策略
                strategy=entity_extraction_config.get("strategy"),
            )
        # 声明提取配置
        claim_extraction_config = values.get("claim_extraction") or {}
        with (
            reader.envvar_prefix(Section.claim_extraction),  # 设置环境变量前缀
            reader.use(claim_extraction_config),  # 使用配置
        ):
            # 获取最大提取数量
            max_gleanings = reader.int(Fragment.max_gleanings)
            max_gleanings = (
                max_gleanings if max_gleanings is not None else defs.CLAIM_MAX_GLEANINGS
            )
            # 创建声明提取模型
            claim_extraction_model = ClaimExtractionConfig(
                enabled=reader.bool(Fragment.enabled) or defs.CLAIM_EXTRACTION_ENABLED,
                llm=hydrate_llm_params(claim_extraction_config, llm_model),
                parallelization=hydrate_parallelization_params(
                    claim_extraction_config, llm_parallelization_model
                ),
                async_mode=hydrate_async_type(claim_extraction_config, async_mode),
                description=reader.str("description") or defs.CLAIM_DESCRIPTION,
                prompt=reader.str("prompt", Fragment.prompt_file),
                max_gleanings=max_gleanings,
                encoding_model=reader.str(Fragment.encoding_model),
            )

        # 社区报告配置
        community_report_config = values.get("community_reports") or {}
        with (
            reader.envvar_prefix(Section.community_reports),  # 设置环境变量前缀
            reader.use(community_report_config),  # 使用配置
        ):
            # 创建社区报告模型
            community_reports_model = CommunityReportsConfig(
                llm=hydrate_llm_params(community_report_config, llm_model),
                parallelization=hydrate_parallelization_params(
                    community_report_config, llm_parallelization_model
                ),
                async_mode=hydrate_async_type(community_report_config, async_mode),
                prompt=reader.str("prompt", Fragment.prompt_file),
                max_length=reader.int(Fragment.max_length)
                           or defs.COMMUNITY_REPORT_MAX_LENGTH,
                max_input_length=reader.int("max_input_length")
                                 or defs.COMMUNITY_REPORT_MAX_INPUT_LENGTH,
            )

        # 摘要描述配置
        summarize_description_config = values.get("summarize_descriptions") or {}
        with (
            reader.envvar_prefix(Section.summarize_descriptions),  # 设置环境变量前缀
            reader.use(values.get("summarize_descriptions")),  # 使用配置
        ):
            # 创建摘要描述模型
            summarize_descriptions_model = SummarizeDescriptionsConfig(
                llm=hydrate_llm_params(summarize_description_config, llm_model),
                parallelization=hydrate_parallelization_params(
                    summarize_description_config, llm_parallelization_model
                ),
                async_mode=hydrate_async_type(summarize_description_config, async_mode),
                prompt=reader.str("prompt", Fragment.prompt_file),
                max_length=reader.int(Fragment.max_length)
                           or defs.SUMMARIZE_DESCRIPTIONS_MAX_LENGTH,
            )

        # 使用 reader.use() 方法读取 cluster_graph 配置
        with reader.use(values.get("cluster_graph")):
            # 创建 ClusterGraphConfig 对象
            cluster_graph_model = ClusterGraphConfig(
                # 读取 max_cluster_size 配置，若未配置则使用默认值
                max_cluster_size=reader.int("max_cluster_size") or defs.MAX_CLUSTER_SIZE
            )

        # 使用 reader.use() 方法读取 local_search 配置，并设置环境变量前缀
        with (
            reader.use(values.get("local_search")),
            reader.envvar_prefix(Section.local_search),
        ):
            # 创建 LocalSearchConfig 对象
            local_search_model = LocalSearchConfig(
                # 读取文本单元比例配置，若未配置则使用默认值
                text_unit_prop=reader.float("text_unit_prop")
                               or defs.LOCAL_SEARCH_TEXT_UNIT_PROP,
                # 读取社区比例配置，若未配置则使用默认值
                community_prop=reader.float("community_prop")
                               or defs.LOCAL_SEARCH_COMMUNITY_PROP,
                # 读取对话历史最大回合数配置，若未配置则使用默认值
                conversation_history_max_turns=reader.int(
                    "conversation_history_max_turns"
                )
                                               or defs.LOCAL_SEARCH_CONVERSATION_HISTORY_MAX_TURNS,
                # 读取映射实体的前 K 个配置，若未配置则使用默认值
                top_k_entities=reader.int("top_k_entities")
                               or defs.LOCAL_SEARCH_TOP_K_MAPPED_ENTITIES,
                # 读取映射关系的前 K 个配置，若未配置则使用默认值
                top_k_relationships=reader.int("top_k_relationships")
                                    or defs.LOCAL_SEARCH_TOP_K_RELATIONSHIPS,
                # 读取温度配置，若未配置则使用默认值
                temperature=reader.float("llm_temperature")
                            or defs.LOCAL_SEARCH_LLM_TEMPERATURE,
                # 读取顶部概率配置，若未配置则使用默认值
                top_p=reader.float("llm_top_p") or defs.LOCAL_SEARCH_LLM_TOP_P,
                # 读取生成的完成次数配置，若未配置则使用默认值
                n=reader.int("llm_n") or defs.LOCAL_SEARCH_LLM_N,
                # 读取最大令牌数配置，若未配置则使用默认值
                max_tokens=reader.int(Fragment.max_tokens)
                           or defs.LOCAL_SEARCH_MAX_TOKENS,
                # 读取 LLM 最大令牌数配置，若未配置则使用默认值
                llm_max_tokens=reader.int("llm_max_tokens")
                               or defs.LOCAL_SEARCH_LLM_MAX_TOKENS,
            )

        # 使用 reader.use() 方法读取 global_search 配置，并设置环境变量前缀
        with (
            reader.use(values.get("global_search")),
            reader.envvar_prefix(Section.global_search),
        ):
            # 创建 GlobalSearchConfig 对象
            global_search_model = GlobalSearchConfig(
                # 读取温度配置，若未配置则使用默认值
                temperature=reader.float("llm_temperature")
                            or defs.GLOBAL_SEARCH_LLM_TEMPERATURE,
                # 读取顶部概率配置，若未配置则使用默认值
                top_p=reader.float("llm_top_p") or defs.GLOBAL_SEARCH_LLM_TOP_P,
                # 读取生成的完成次数配置，若未配置则使用默认值
                n=reader.int("llm_n") or defs.GLOBAL_SEARCH_LLM_N,
                # 读取最大令牌数配置，若未配置则使用默认值
                max_tokens=reader.int(Fragment.max_tokens)
                           or defs.GLOBAL_SEARCH_MAX_TOKENS,
                # 读取数据最大令牌数配置，若未配置则使用默认值
                data_max_tokens=reader.int("data_max_tokens")
                                or defs.GLOBAL_SEARCH_DATA_MAX_TOKENS,
                # 读取映射最大令牌数配置，若未配置则使用默认值
                map_max_tokens=reader.int("map_max_tokens")
                               or defs.GLOBAL_SEARCH_MAP_MAX_TOKENS,
                # 读取降序最大令牌数配置，若未配置则使用默认值
                reduce_max_tokens=reader.int("reduce_max_tokens")
                                  or defs.GLOBAL_SEARCH_REDUCE_MAX_TOKENS,
                # 读取并发度配置，若未配置则使用默认值
                concurrency=reader.int("concurrency") or defs.GLOBAL_SEARCH_CONCURRENCY,
            )

        # 读取编码模型配置，若未配置则使用默认值
        encoding_model = reader.str(Fragment.encoding_model) or defs.ENCODING_MODEL
        skip_workflows = reader.list("skip_workflows") or []

    return GraphRagConfig(
        root_dir=root_dir,
        llm=llm_model,
        parallelization=llm_parallelization_model,
        async_mode=async_mode,
        embeddings=embeddings_model,
        embed_graph=embed_graph_model,
        reporting=reporting_model,
        storage=storage_model,
        cache=cache_model,
        input=input_model,
        chunks=chunks_model,
        snapshots=snapshots_model,
        entity_extraction=entity_extraction_model,
        claim_extraction=claim_extraction_model,
        community_reports=community_reports_model,
        summarize_descriptions=summarize_descriptions_model,
        umap=umap_model,
        cluster_graph=cluster_graph_model,
        encoding_model=encoding_model,
        skip_workflows=skip_workflows,
        local_search=local_search_model,
        global_search=global_search_model,
    )


class Fragment(str, Enum):
    """
    配置片段枚举类

    该枚举类定义了各种配置片段的名称，用于在配置文件中引用。
    """

    # API 基础 URL
    api_base = "API_BASE"

    # API 密钥
    api_key = "API_KEY"

    # API 版本
    api_version = "API_VERSION"

    # API 组织
    api_organization = "API_ORGANIZATION"

    # API 代理
    api_proxy = "API_PROXY"

    # 异步模式
    async_mode = "ASYNC_MODE"

    # 基础目录
    base_dir = "BASE_DIR"

    # 认知服务端点
    cognitive_services_endpoint = "COGNITIVE_SERVICES_ENDPOINT"

    # 并发请求
    concurrent_requests = "CONCURRENT_REQUESTS"

    # 连接字符串
    conn_string = "CONNECTION_STRING"

    # 容器名称
    container_name = "CONTAINER_NAME"

    # 仓库 ID
    compartment_id = "COMPARTMENT_ID"

    # 配置文件
    config_profile = "CONFIG_PROFILE"

    # 部署名称
    deployment_name = "DEPLOYMENT_NAME"

    # 描述
    description = "DESCRIPTION"

    # 启用状态
    enabled = "ENABLED"

    # 编码
    encoding = "ENCODING"

    # 编码模型
    encoding_model = "ENCODING_MODEL"

    # 端点
    endpoint = "ENDPOINT"

    # 文件类型
    file_type = "FILE_TYPE"

    # 最大获取数量
    max_gleanings = "MAX_GLEANINGS"

    # 最大长度
    max_length = "MAX_LENGTH"

    # 最大重试次数
    max_retries = "MAX_RETRIES"

    # 最大重试等待时间
    max_retry_wait = "MAX_RETRY_WAIT"

    # 最大令牌数
    max_tokens = "MAX_TOKENS"

    # 温度
    temperature = "TEMPERATURE"

    # 顶部 P
    top_p = "TOP_P"

    # 顶部 K
    top_k = "TOP_K"

    # N
    n = "N"

    # 模型
    model = "MODEL"

    # 模型 ID
    model_id = "MODEL_ID"

    # 模型支持 JSON
    model_supports_json = "MODEL_SUPPORTS_JSON"

    # 提示文件
    prompt_file = "PROMPT_FILE"

    # 请求超时
    request_timeout = "REQUEST_TIMEOUT"

    # 每分钟请求数
    rpm = "REQUESTS_PER_MINUTE"

    # 睡眠推荐
    sleep_recommendation = "SLEEP_ON_RATE_LIMIT_RECOMMENDATION"

    # 存储帐户 Blob URL
    storage_account_blob_url = "STORAGE_ACCOUNT_BLOB_URL"

    # 线程数
    thread_count = "THREAD_COUNT"

    # 线程延迟
    thread_stagger = "THREAD_STAGGER"

    # 每分钟令牌数
    tpm = "TOKENS_PER_MINUTE"

    # 类型
    type = "TYPE"


# 配置节枚举类
class Section(str, Enum):
    """
    配置节枚举类。

    该枚举类定义了配置文件中的各个节。
    """

    # 基础配置节
    base = "BASE"
    # 缓存配置节
    cache = "CACHE"
    # 数据块配置节
    chunk = "CHUNK"
    # 声明提取配置节
    claim_extraction = "CLAIM_EXTRACTION"
    # 社区报告配置节
    community_reports = "COMMUNITY_REPORTS"
    # 嵌入配置节
    embedding = "EMBEDDING"
    # 实体提取配置节
    entity_extraction = "ENTITY_EXTRACTION"
    # GraphRAG 配置节
    graphrag = "GRAPHRAG"
    # 输入配置节
    input = "INPUT"
    # 语言模型配置节
    llm = "LLM"
    # Node2Vec 配置节
    node2vec = "NODE2VEC"
    # 报告配置节
    reporting = "REPORTING"
    # 快照配置节
    snapshot = "SNAPSHOT"
    # 存储配置节
    storage = "STORAGE"
    # 描述总结配置节
    summarize_descriptions = "SUMMARIZE_DESCRIPTIONS"
    # UMAP 配置节
    umap = "UMAP"
    # 本地搜索配置节
    local_search = "LOCAL_SEARCH"
    # 全局搜索配置节
    global_search = "GLOBAL_SEARCH"


# 判断是否为 Azure 语言模型
def _is_azure(llm_type: LLMType | None) -> bool:
    """
    判断是否为 Azure 语言模型。

    Args:
        llm_type: 语言模型类型

    Returns:
        bool: 是否为 Azure 语言模型
    """
    return (
        llm_type == LLMType.AzureOpenAI
        or llm_type == LLMType.AzureOpenAIChat
        or llm_type == LLMType.AzureOpenAIEmbedding
    )


# 创建环境变量
def _make_env(root_dir: str) -> Env:
    """
    创建环境变量。

    Args:
        root_dir: 根目录

    Returns:
        Env: 环境变量对象
    """
    # 读取 .env 文件
    read_dotenv(root_dir)
    # 创建环境变量对象
    env = Env(expand_vars=True)
    # 读取环境变量
    env.read_env()
    return env


# 替换环境变量中的占位符
def _token_replace(data: dict):
    """
    替换环境变量中的占位符。

    Args:
        data: 环境变量字典
    """
    # 遍历环境变量字典
    for key, value in data.items():
        # 如果值是字典，则递归替换
        if isinstance(value, dict):
            _token_replace(value)
        # 如果值是字符串，则替换占位符
        elif isinstance(value, str):
            data[key] = os.path.expandvars(value)
