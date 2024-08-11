# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""
配置参数化设置
================

该模块定义了存储配置的默认设置
"""

from pydantic import BaseModel, Field

import graphrag.config.defaults as defs
from graphrag.config.enums import StorageType


class StorageConfig(BaseModel):
    """
    存储配置的默认设置
    --------------------

    该类定义了存储配置的基本结构，包括存储类型、基本目录、连接字符串、容器名称和存储账户 Blob URL。

    Attributes:
        type (StorageType): 存储类型
        base_dir (str): 基本目录
        connection_string (str | None): 连接字符串
        container_name (str | None): 容器名称
        storage_account_blob_url (str | None): 存储账户 Blob URL
    """

    # 存储类型
    type: StorageType = Field(
        description="要使用的存储类型",  # 存储类型的描述
        default=defs.STORAGE_TYPE  # 默认值
    )

    # 基本目录
    base_dir: str = Field(
        description="存储的基本目录",  # 基本目录的描述
        default=defs.STORAGE_BASE_DIR  # 默认值
    )

    # 连接字符串
    connection_string: str | None = Field(
        description="要使用的连接字符串",  # 连接字符串的描述
        default=None  # 默认值
    )

    # 容器名称
    container_name: str | None = Field(
        description="要使用的容器名称",  # 容器名称的描述
        default=None  # 默认值
    )

    # 存储账户 Blob URL
    storage_account_blob_url: str | None = Field(
        description="要使用的存储账户 Blob URL",  # 存储账户 Blob URL 的描述
        default=None  # 默认值
    )
