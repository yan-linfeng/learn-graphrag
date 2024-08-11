# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""
存储配置的默认参数设置。

该模块包含了存储配置的参数设置，包括类型、基本目录、连接字符串、容器名称和存储账户 Blob URL。
"""

from typing_extensions import NotRequired, TypedDict

from graphrag.config.enums import StorageType


class StorageConfigInput(TypedDict):
    """
    存储配置的默认配置节。

    该类定义了存储配置的参数，包括类型、基本目录、连接字符串、容器名称和存储账户 Blob URL。
    """

    type: NotRequired[StorageType | str | None]  # 存储类型，允许传入 StorageType 枚举值或字符串
    base_dir: NotRequired[str | None]  # 基本目录，存储文件的根目录
    connection_string: NotRequired[str | None]  # 连接字符串，用于连接到存储服务
    container_name: NotRequired[str | None]  # 容器名称，存储文件的容器名称
    storage_account_blob_url: NotRequired[str | None]  # 存储账户 Blob URL，存储文件的 Blob URL
