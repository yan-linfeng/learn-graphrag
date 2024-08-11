# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""默认配置的参数化设置"""

# 导入 Pydantic 的 BaseModel 和 Field 类
from pydantic import BaseModel, Field

# 导入 graphrag.config.defaults 模块中的默认值
import graphrag.config.defaults as defs
# 导入 graphrag.config.enums 模块中的 ReportingType 枚举类型
from graphrag.config.enums import ReportingType


class ReportingConfig(BaseModel):
    """
    Reporting 配置的默认设置

    该类定义了 Reporting 配置的默认值，包括报告类型、基本目录、连接字符串、容器名称和存储账户 Blob URL。
    """

    # 报告类型
    type: ReportingType = Field(
        description="要使用的报告类型",  # 报告类型的描述
        default=defs.REPORTING_TYPE  # 默认值
    )
    # 基本目录
    base_dir: str = Field(
        description="报告的基本目录",  # 基本目录的描述
        default=defs.REPORTING_BASE_DIR  # 默认值
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
