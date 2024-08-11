# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""
默认配置的参数化设置。

该模块包含了缓存配置的参数设置，包括类型、基本目录、连接字符串、容器名称和存储账户 Blob URL。
"""

from pydantic import BaseModel, Field

import graphrag.config.defaults as defs
from graphrag.config.enums import CacheType


class CacheConfig(BaseModel):
    """
    缓存配置的默认配置节。

    该类定义了缓存配置的参数，包括类型、基本目录、连接字符串、容器名称和存储账户 Blob URL。
    """

    # 缓存类型
    type: CacheType = Field(
        description="要使用的缓存类型。", default=defs.CACHE_TYPE
    )
    # 缓存的基本目录
    base_dir: str = Field(
        description="缓存的基本目录。", default=defs.CACHE_BASE_DIR
    )
    # 缓存连接字符串
    connection_string: str | None = Field(
        description="要使用的缓存连接字符串。", default=None
    )
    # 缓存容器名称
    container_name: str | None = Field(
        description="要使用的缓存容器名称。", default=None
    )
    # 存储账户 Blob URL
    storage_account_blob_url: str | None = Field(
        description="要使用的存储账户 Blob URL。", default=None
    )