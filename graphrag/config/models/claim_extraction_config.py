# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""参数化设置默认配置"""

from pathlib import Path

from pydantic import Field

import graphrag.config.defaults as defs

from .llm_config import LLMConfig


class ClaimExtractionConfig(LLMConfig):
    """
    声明提取配置节。

    本节定义了声明提取的相关参数，包括启用状态、提示语、描述、最大获取数、策略和编码模型。
    """

    enabled: bool = Field(
        description="是否启用声明提取。",
    )
    # 提示语，用于声明提取
    prompt: str | None = Field(
        description="声明提取提示语。", default=None
    )
    # 声明描述
    description: str = Field(
        description="声明描述。",
        default=defs.CLAIM_DESCRIPTION,
    )
    # 最大获取数
    max_gleanings: int = Field(
        description="最大获取的实体数量。",
        default=defs.CLAIM_MAX_GLEANINGS,
    )
    # 策略，用于覆盖默认策略
    strategy: dict | None = Field(
        description="覆盖默认策略。", default=None
    )
    # 编码模型
    encoding_model: str | None = Field(
        default=None, description="编码模型。"
    )

    def resolved_strategy(self, root_dir: str, encoding_model: str) -> dict:
        """
        获取解析后的声明提取策略。

        如果未指定策略，则使用默认策略。

        :param root_dir: 根目录
        :param encoding_model: 编码模型
        :return: 解析后的策略
        """
        from graphrag.index.verbs.covariates.extract_covariates import (
            ExtractClaimsStrategyType,
        )

        # 如果未指定策略，则使用默认策略
        return self.strategy or {
            "type": ExtractClaimsStrategyType.graph_intelligence,
            "llm": self.llm.model_dump(),
            **self.parallelization.model_dump(),
            # 提取提示语
            "extraction_prompt": (Path(root_dir) / self.prompt)
            .read_bytes()
            .decode(encoding="utf-8")
            if self.prompt
            else None,
            # 声明描述
            "claim_description": self.description,
            # 最大获取数
            "max_gleanings": self.max_gleanings,
            # 编码模型
            "encoding_name": self.encoding_model or encoding_model,
        }
