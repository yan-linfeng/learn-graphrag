# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""LLM Parameters model."""

from pydantic import BaseModel, ConfigDict, Field

import graphrag.config.defaults as defs
from graphrag.config.enums import LLMType


class LLMParameters(BaseModel):
    """
    LLM Parameters model.

    本类定义了LLM服务的参数模型，包括API密钥、模型类型、模型ID、最大令牌数等。
    """

    # 模型配置
    model_config = ConfigDict(protected_namespaces=(), extra="allow")

    # API密钥
    api_key: str | None = Field(
        description="API密钥",
        default=None,
    )
    """
    API密钥用于身份验证，请确保填写正确的密钥。
    """

    # 模型类型
    type: LLMType = Field(
        description="模型类型",
        default=defs.LLM_TYPE
    )
    """
    模型类型决定了LLM服务的行为，请选择合适的类型。
    """

    # 模型ID
    model: str = Field(description="模型ID", default=defs.LLM_MODEL)
    """
    模型ID用于唯一标识LLM模型，请确保填写正确的ID。
    """

    # 模型ID
    model_id: str = Field(description="模型ID", default=defs.LLM_MODEL_ID)
    """
    模型ID用于唯一标识LLM模型，请确保填写正确的ID。
    """

    # 最大令牌数
    max_tokens: int | None = Field(
        description="最大令牌数",
        default=defs.LLM_MAX_TOKENS,
    )
    """
    最大令牌数决定了LLM服务可以处理的最大文本长度，请根据需要调整。
    """

    # 温度
    temperature: float | None = Field(
        description="温度",
        default=defs.LLM_TEMPERATURE,
    )
    """
    温度决定了LLM服务的随机性，请根据需要调整。
    """

    # top-p值
    top_p: float | None = Field(
        description="top-p值",
        default=defs.LLM_TOP_P,
    )
    """
    top-p值决定了LLM服务的随机性，请根据需要调整。
    """

    # top-k值
    top_k: int | None = Field(
        description="top-k值",
        default=defs.LLM_TOP_K,
    )
    """
    top-k值决定了LLM服务的随机性，请根据需要调整。
    """

    # 完成数
    n: int | None = Field(
        description="完成数",
        default=defs.LLM_N,
    )
    """
    完成数决定了LLM服务可以生成的最大完成数，请根据需要调整。
    """

    # 请求超时
    request_timeout: float = Field(
        description="请求超时",
        default=defs.LLM_REQUEST_TIMEOUT
    )
    """
    请求超时决定了LLM服务的超时时间，请根据需要调整。
    """

    # API基础URL
    api_base: str | None = Field(
        description="API基础URL",
        default=None
    )
    """
    API基础URL用于唯一标识LLM服务的API，请确保填写正确的URL。
    """

    # API端点URL
    endpoint: str | None = Field(
        description="API端点URL",
        default=None
    )
    """
    API端点URL用于唯一标识LLM服务的API端点，请确保填写正确的URL。
    """

    # API版本
    api_version: str | None = Field(
        description="API版本",
        default=None
    )
    """
    API版本决定了LLM服务的API版本，请根据需要调整。
    """

    # 组织
    organization: str | None = Field(
        description="组织",
        default=None
    )
    """
    组织用于唯一标识LLM服务的组织，请确保填写正确的组织。
    """

    # 配置-config_file
    config_file: str | None = Field(
        description="配置-config_file",
        default="~/.oci/config"
    )
    """
    配置-profile用于唯一标识LLM服务的配置-profile，请确保填写正确的配置-profile。
    """

    # 配置-profile
    config_profile: str | None = Field(
        description="配置-profile",
        default=None
    )

    # comparment ID
    compartment_id: str | None = Field(
        description="comparment ID",
        default="DEFAULT"
    )
    """
    comparment ID用于唯一标识LLM服务的comparment，请确保填写正确的comparment ID。
    """

    # 代理服务器配置
    proxy: str | None = Field(
        description="LLM 服务使用的代理服务器地址",
        default=None
    )
    """
    代理服务器配置用于指定 LLM 服务使用的代理服务器地址。
    """

    # 认知服务端点配置
    cognitive_services_endpoint: str | None = Field(
        description="访问认知服务的端点地址",
        default=None
    )
    """
    认知服务端点配置用于指定访问认知服务的端点地址。
    """

    # 部署名称配置
    deployment_name: str | None = Field(
        description="LLM 服务使用的部署名称",
        default=None
    )
    """
    部署名称配置用于指定 LLM 服务使用的部署名称。
    """

    # 模型支持 JSON 输出模式配置
    model_supports_json: bool | None = Field(
        description="模型是否支持 JSON 输出模式",
        default=None
    )
    """
    模型支持 JSON 输出模式配置用于指定模型是否支持 JSON 输出模式。
    """

    # 每分钟令牌数配置
    tokens_per_minute: int = Field(
        description="LLM 服务每分钟使用的令牌数",
        default=defs.LLM_TOKENS_PER_MINUTE,
    )
    """
    每分钟令牌数配置用于指定 LLM 服务每分钟使用的令牌数。
    """

    # 每分钟请求数配置
    requests_per_minute: int = Field(
        description="LLM 服务每分钟使用的请求数",
        default=defs.LLM_REQUESTS_PER_MINUTE,
    )
    """
    每分钟请求数配置用于指定 LLM 服务每分钟使用的请求数。
    """

    # 最大重试次数配置
    max_retries: int = Field(
        description="LLM 服务的最大重试次数",
        default=defs.LLM_MAX_RETRIES,
    )
    """
    最大重试次数配置用于指定 LLM 服务的最大重试次数。
    """

    # 最大重试等待时间配置
    max_retry_wait: float = Field(
        description="LLM 服务的最大重试等待时间",
        default=defs.LLM_MAX_RETRY_WAIT,
    )
    """
    最大重试等待时间配置用于指定 LLM 服务的最大重试等待时间。
    """

    # 是否在速率限制推荐时睡眠配置
    sleep_on_rate_limit_recommendation: bool = Field(
        description="是否在速率限制推荐时睡眠",
        default=defs.LLM_SLEEP_ON_RATE_LIMIT_RECOMMENDATION,
    )
    """
    是否在速率限制推荐时睡眠配置用于指定是否在速率限制推荐时睡眠。
    """

    # 是否使用并发请求配置
    concurrent_requests: int = Field(
        description="是否使用并发请求",
        default=defs.LLM_CONCURRENT_REQUESTS,
    )
    """
    是否使用并发请求配置用于指定是否使用并发请求。
    """
