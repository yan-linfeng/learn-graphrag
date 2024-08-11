# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""语言模型（LLM）参数模型"""

from typing_extensions import NotRequired, TypedDict


class ParallelizationParametersInput(TypedDict):
    """
    并行化参数输入模型

    该模型定义了并行化参数的输入结构，包括 stagger 和 num_threads 两个参数。
    """

    # 并行化延迟参数（可选）
    stagger: NotRequired[float | str | None]  # 延迟值，可以是浮点数、字符串或 None

    # 并行化线程数参数（可选）
    num_threads: NotRequired[int | str | None]  # 线程数，可以是整数、字符串或 None
