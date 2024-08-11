# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""默认配置的参数化设置。"""

from typing_extensions import NotRequired, TypedDict


class ClusterGraphConfigInput(TypedDict):
    """
    聚类图配置部分。

    该类定义了聚类图的配置参数，包括最大聚类大小和策略。
    """

    # 最大聚类大小，允许为 None
    max_cluster_size: NotRequired[int | None]

    # 聚类策略，允许为 None
    strategy: NotRequired[dict | None]
