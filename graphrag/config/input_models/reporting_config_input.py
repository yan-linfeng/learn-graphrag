# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""配置参数的默认设置。"""

# 从 typing_extensions 模块导入 NotRequired 和 TypedDict 类型
from typing_extensions import NotRequired, TypedDict

# 从 graphrag.config.enums 模块导入 ReportingType 枚举类型
from graphrag.config.enums import ReportingType


class ReportingConfigInput(TypedDict):
    """
    报告配置的默认配置节。

    该类定义了报告配置的参数，包括类型、基本目录、连接字符串、容器名称和存储账户 Blob URL。
    """

    # 报告类型（可选）
    type: NotRequired[ReportingType | str | None]  # 类型： ReportingType 或字符串或 None

    # 基本目录（可选）
    base_dir: NotRequired[str | None]  # 基本目录：字符串或 None

    # 连接字符串（可选）
    connection_string: NotRequired[str | None]  # 连接字符串：字符串或 None

    # 容器名称（可选）
    container_name: NotRequired[str | None]  # 容器名称：字符串或 None

    # 存储账户 Blob URL（可选）
    storage_account_blob_url: NotRequired[str | None]  # 存储账户 Blob URL：字符串或 None
