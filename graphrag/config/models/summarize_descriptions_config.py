# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""
配置默认设置的参数化配置。

本模块定义了描述摘要配置的参数化设置，包括描述摘要提示、最大长度和策略。
"""

from pathlib import Path

from pydantic import Field

import graphrag.config.defaults as defs

from .llm_config import LLMConfig


class SummarizeDescriptionsConfig(LLMConfig):
    """
    描述摘要配置。

    本类定义了描述摘要配置的参数化设置，包括描述摘要提示、最大长度和策略。
    """

    # 描述摘要提示
    prompt: str | None = Field(
        description="描述摘要提示，用于描述摘要的提示信息。", default=None
    )
    # 描述摘要最大长度
    max_length: int = Field(
        description="描述摘要最大长度，用于限制描述摘要的长度。",
        default=defs.SUMMARIZE_DESCRIPTIONS_MAX_LENGTH,
    )
    # 策略
    strategy: dict | None = Field(
        description="策略，用于描述摘要的策略。", default=None
    )

    def resolved_strategy(self, root_dir: str) -> dict:
        """
        获取描述摘要策略。

        本方法返回描述摘要策略，包括描述摘要类型、LLM模型、并行化配置和描述摘要提示。

        :param root_dir: 根目录
        :return: 描述摘要策略
        """
        from graphrag.index.verbs.entities.summarize import SummarizeStrategyType

        # 如果策略存在，则返回策略
        if self.strategy:
            return self.strategy
        # 否则，返回默认策略
        else:
            return {
                "type": SummarizeStrategyType.graph_intelligence,
                "llm": self.llm.model_dump(),
                **self.parallelization.model_dump(),
                "summarize_prompt": (Path(root_dir) / self.prompt)
                .read_bytes()
                .decode(encoding="utf-8")
                if self.prompt
                else None,
                "max_summary_length": self.max_length,
            }