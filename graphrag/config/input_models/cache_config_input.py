# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""
默认配置的参数化设置。
"""

from typing_extensions import NotRequired, TypedDict

from graphrag.config.enums import CacheType


class CacheConfigInput(TypedDict):
    """
    缓存配置的默认配置节。

    该配置节定义了缓存的基本设置，包括缓存类型、基本目录、连接字符串、容器名称和存储账户 Blob URL。
    """

    # 缓存类型（可选）
    type: NotRequired[CacheType | str | None]  # 缓存类型，可以是 CacheType 或字符串

    # 基本目录（可选）
    base_dir: NotRequired[str | None]  # 缓存的基本目录

    # 连接字符串（可选）
    connection_string: NotRequired[str | None]  # 缓存的连接字符串

    # 容器名称（可选）
    container_name: NotRequired[str | None]  # 缓存的容器名称

    # 存储账户 Blob URL（可选）
    storage_account_blob_url: NotRequired[str | None]  # 缓存的存储账户 Blob URL
