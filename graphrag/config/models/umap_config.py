# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""默认配置的参数化设置"""

# 导入 Pydantic 的 BaseModel 和 Field 类
from pydantic import BaseModel, Field

# 导入 graphrag.config.defaults 模块中的默认值
import graphrag.config.defaults as defs


class UmapConfig(BaseModel):
    """
    UMAP 配置部分。

    本类定义了 UMAP 的配置设置，包括是否启用 UMAP。
    """

    # 是否启用 UMAP 的标志
    enabled: bool = Field(
        description="是否启用 UMAP 的标志。",
        default=defs.UMAP_ENABLED,
    )
