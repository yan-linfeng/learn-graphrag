# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""
A module containing cluster_graph, apply_clustering and run_layout methods definition.
"""

import logging
from enum import Enum
from random import Random
from typing import Any, cast, List, Tuple

import networkx as nx
import pandas as pd
from datashaper import TableContainer, VerbCallbacks, VerbInput, progress_iterable, verb

from graphrag.index.utils import gen_uuid, load_graph

from .typing import Communities

log = logging.getLogger(__name__)


@verb(name="cluster_graph")
def cluster_graph(
    input: VerbInput,
    callbacks: VerbCallbacks,
    strategy: dict[str, Any],
    column: str,
    to: str,
    level_to: str | None = None,
    **_kwargs,
) -> TableContainer:
    """
    将图进行层次聚类。图应该是以graphml格式存在的。该方法会输出一个包含聚类后的图的新列，以及一个包含图的层级的新列。

    ## 使用方法
    ```yaml
    verb: cluster_graph
    args:
        column: entity_graph # 包含图的列的名称，应该是graphml格式的图
        to: clustered_graph # 输出聚类后的图的列的名称
        level_to: level # 输出图的层级的列的名称
        strategy: <策略配置> # 参与聚类的策略配置，详见“策略”部分
    ```

    ## 策略
    cluster graph verb使用一个策略来聚类图。该策略是一个定义策略的JSON对象。以下是可用的策略：

    ### leiden
    该策略使用leiden算法来聚类图。策略配置如下：
    ```yaml
    strategy:
        type: leiden
        max_cluster_size: 10 # 可选，最大聚类大小，默认为10
        use_lcc: true # 可选，如果使用leiden算法的最大的连通组件，默认为true
        seed: 0xDEADBEEF # 可选，leiden算法的随机种子，默认为0xDEADBEEF
        levels: [0, 1] # 可选，输出的层级，默认为所有检测到的层级
    ```
    """
    output_df = cast(pd.DataFrame, input.get_input())
    results = output_df[column].apply(lambda graph: run_layout(strategy, graph))

    community_map_to = "communities"
    output_df[community_map_to] = results

    level_to = level_to or f"{to}_level"
    output_df[level_to] = output_df.apply(
        lambda x: list({level for level, _, _ in x[community_map_to]}), axis=1
    )
    output_df[to] = [None] * len(output_df)

    num_total = len(output_df)

    # 遍历每一行
    graph_level_pairs_column: List[List[Tuple[int, str]]] = []
    for _, row in progress_iterable(
        output_df.iterrows(), callbacks.progress, num_total
    ):
        levels = row[level_to]
        graph_level_pairs: List[Tuple[int, str]] = []

        # 对于每个层级，获取图并添加到列表中
        for level in levels:
            graph = "\n".join(
                nx.generate_graphml(
                    apply_clustering(
                        cast(str, row[column]),
                        cast(Communities, row[community_map_to]),
                        level,
                    )
                )
            )
            graph_level_pairs.append((level, graph))
        graph_level_pairs_column.append(graph_level_pairs)
    output_df[to] = graph_level_pairs_column

    # 将（层级，图）对分解成单独的行
    output_df = output_df.explode(to, ignore_index=True)

    # 将（层级，图）对分解成单独的列
    # TODO: 可能有更好的方式来实现
    output_df[[level_to, to]] = pd.DataFrame(
        output_df[to].tolist(), index=output_df.index
    )

    # 清理社区映射
    output_df.drop(columns=[community_map_to], inplace=True)

    return TableContainer(table=output_df)


# TODO: 应该支持 str | nx.Graph 作为 graphml 参数
def apply_clustering(
    graphml: str, communities: Communities, level=0, seed=0xF001
) -> nx.Graph:
    """
    应用聚类到图形中。

    Args:
        graphml (str): 图形的 GraphML 字符串。
        communities (Communities): 聚类结果。
        level (int, optional): 聚类层级。 Defaults to 0.
        seed (int, optional): 随机数种子。 Defaults to 0xF001.

    Returns:
        nx.Graph: 聚类后的图形。
    """
    # 初始化随机数生成器
    random = Random(seed)  # noqa S311

    # 解析 GraphML 字符串为图形
    graph = nx.parse_graphml(graphml)

    # 遍历聚类结果
    for community_level, community_id, nodes in communities:
        # 如果当前层级与聚类层级匹配
        if level == community_level:
            # 遍历节点
            for node in nodes:
                # 设置节点的聚类 ID 和层级
                graph.nodes[node]["cluster"] = community_id
                graph.nodes[node]["level"] = level

    # 添加节点度
    for node_degree in graph.degree:
        # 设置节点的度
        graph.nodes[str(node_degree[0])]["degree"] = int(node_degree[1])

    # 添加节点 UUID 和增量 ID（用于最终报告）
    for index, node in enumerate(graph.nodes()):
        # 设置节点的 UUID 和增量 ID
        graph.nodes[node]["human_readable_id"] = index
        graph.nodes[node]["id"] = str(gen_uuid(random))

    # 添加边 ID
    for index, edge in enumerate(graph.edges()):
        # 设置边的 ID 和增量 ID
        graph.edges[edge]["id"] = str(gen_uuid(random))
        graph.edges[edge]["human_readable_id"] = index
        graph.edges[edge]["level"] = level

    # 返回聚类后的图形
    return graph


class GraphCommunityStrategyType(str, Enum):
    """
    图形聚类策略类型。
    """
    leiden = "leiden"

    def __repr__(self):
        """
        获取字符串表示。
        """
        return f'"{self.value}"'


def run_layout(
    strategy: dict[str, Any], graphml_or_graph: str | nx.Graph
) -> Communities:
    """
    运行布局算法。

    Args:
        strategy (dict[str, Any]): 聚类策略。
        graphml_or_graph (str | nx.Graph): 图形或 GraphML 字符串。

    Returns:
        Communities: 聚类结果。
    """
    # 加载图形
    graph = load_graph(graphml_or_graph)

    # 如果图形为空
    if len(graph.nodes) == 0:
        # 输出警告信息
        log.warning("图形为空")
        # 返回空聚类结果
        return []

    # 初始化聚类结果
    clusters: dict[int, dict[str, list[str]]] = {}

    # 获取聚类策略类型
    strategy_type = strategy.get("type", GraphCommunityStrategyType.leiden)

    # 匹配聚类策略类型
    match strategy_type:
        case GraphCommunityStrategyType.leiden:
            # 导入 Leiden 聚类算法
            from .strategies.leiden import run as run_leiden

            # 运行 Leiden 聚类算法
            clusters = run_leiden(graph, strategy)
        case _:
            # 输出错误信息
            msg = f"未知聚类策略 {strategy_type}"
            # 抛出 ValueError
            raise ValueError(msg)

    # 初始化聚类结果列表
    results: Communities = []

    # 遍历聚类结果
    for level in clusters:
        # 遍历聚类 ID 和节点列表
        for cluster_id, nodes in clusters[level].items():
            # 添加聚类结果到列表
            results.append((level, cluster_id, nodes))

    # 返回聚类结果
    return results
