# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""LLM 参数模型"""

from typing_extensions import NotRequired, TypedDict

from graphrag.config.enums import LLMType


class LLMParametersInput(TypedDict):
    """
    LLM 参数模型。

    该模型定义了 LLM 的输入参数，包括 API 密钥、类型、模型、模型 ID 等。
    """

    # API 密钥（可选）
    api_key: NotRequired[str | None]
    """
    API 密钥，用于身份验证。
    """

    # 类型（可选）
    type: NotRequired[LLMType | str | None]
    """
    LLM 类型，例如文本生成、语义搜索等。
    """

    # 模型（可选）
    model: NotRequired[str | None]
    """
    LLM 模型名称。
    """

    # 模型 ID（可选）
    model_id: NotRequired[str | None]
    """
    LLM 模型 ID。
    """

    # 最大令牌数（可选）
    max_tokens: NotRequired[int | str | None]
    """
    LLM 输入的最大令牌数。
    """

    # 请求超时（可选）
    request_timeout: NotRequired[float | str | None]
    """
    LLM 请求的超时时间。
    """

    # API 基础 URL（可选）
    api_base: NotRequired[str | None]
    """
    LLM API 的基础 URL。
    """

    # API 端点（可选）
    endpoint: NotRequired[str | None]
    """
    LLM API 的端点。
    """

    # API 版本（可选）
    api_version: NotRequired[str | None]
    """
    LLM API 的版本。
    """

    # 组织（可选）
    organization: NotRequired[str | None]
    """
    LLM 所属的组织。
    """

    # 配置文件（可选）
    config_profile: NotRequired[str | None]
    """
    LLM 的配置文件。
    """

    # 仓库 ID（可选）
    compartment_id: NotRequired[str | None]
    """
    LLM 所属的仓库 ID。
    """

    # 代理（可选）
    proxy: NotRequired[str | None]
    """
    LLM 的代理。
    """

    # 认知服务端点（可选）
    cognitive_services_endpoint: NotRequired[str | None]
    """
    LLM 的认知服务端点。
    """

    # 部署名称（可选）
    deployment_name: NotRequired[str | None]
    """
    LLM 的部署名称。
    """

    # 模型支持 JSON（可选）
    model_supports_json: NotRequired[bool | str | None]
    """
    LLM 模型是否支持 JSON。
    """

    # 每分钟令牌数（可选）
    tokens_per_minute: NotRequired[int | str | None]
    """
    LLM 每分钟的令牌数。
    """

    # 每分钟请求数（可选）
    requests_per_minute: NotRequired[int | str | None]
    """
    LLM 每分钟的请求数。
    """

    # 最大重试次数（可选）
    max_retries: NotRequired[int | str | None]
    """
    LLM 的最大重试次数。
    """

    # 最大重试等待时间（可选）
    max_retry_wait: NotRequired[float | str | None]
    """
    LLM 的最大重试等待时间。
    """

    # 睡眠在速率限制推荐（可选）
    sleep_on_rate_limit_recommendation: NotRequired[bool | str | None]
    """
    LLM 是否在速率限制推荐时睡眠。
    """

    # 并发请求（可选）
    concurrent_requests: NotRequired[int | str | None]
    """
    LLM 的并发请求数。
    """
