# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing load_workflows, create_workflow, _get_steps_for_workflow and _remove_disabled_steps methods definition."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from datashaper import Workflow

from graphrag.index.errors import (
    NoWorkflowsDefinedError,
    UndefinedWorkflowError,
    UnknownWorkflowError,
)
from graphrag.index.utils import topological_sort

from .default_workflows import default_workflows as _default_workflows
from .typing import VerbDefinitions, WorkflowDefinitions, WorkflowToRun

if TYPE_CHECKING:
    from graphrag.index.config import (
        PipelineWorkflowConfig,
        PipelineWorkflowReference,
        PipelineWorkflowStep,
    )

anonymous_workflow_count = 0

VerbFn = Callable[..., Any]
log = logging.getLogger(__name__)


class LoadWorkflowResult(NamedTuple):
    """A workflow loading result object."""

    workflows: list[WorkflowToRun]
    """The loaded workflow names in the order they should be run."""

    dependencies: dict[str, list[str]]
    """A dictionary of workflow name to workflow dependencies."""


def load_workflows(
    workflows_to_load: list[PipelineWorkflowReference],
    additional_verbs: VerbDefinitions | None = None,
    additional_workflows: WorkflowDefinitions | None = None,
    memory_profile: bool = False,
) -> LoadWorkflowResult:
    """
    加载指定的工作流

    Args:
        - workflows_to_load - 要加载的工作流
        - additional_verbs - 自定义的动词列表
        - additional_workflows - 自定义的工作流列表
    Returns:
        - output[0] - 加载的工作流名称，按运行顺序排列
        - output[1] - 工作流名称到依赖关系的字典
    """
    # 工作流图，存储工作流名称到 WorkflowToRun 对象的映射
    workflow_graph: dict[str, WorkflowToRun] = {}

    # 全局变量，用于记录匿名工作流的数量
    global anonymous_workflow_count
    for reference in workflows_to_load:
        # 获取工作流名称，如果为空，则使用匿名工作流名称
        print(f"{reference.name=}")
        name = reference.name
        is_anonymous = name is None or name.strip() == ""
        if is_anonymous:
            name = f"Anonymous Workflow {anonymous_workflow_count}"
            anonymous_workflow_count += 1
        name = cast(str, name)

        # 获取工作流配置
        config = reference.config
        # 创建工作流对象
        workflow = create_workflow(
            name or "MISSING NAME!",
            reference.steps,
            config,
            additional_verbs,
            additional_workflows,
        )
        # 将工作流对象添加到工作流图中
        workflow_graph[name] = WorkflowToRun(workflow, config=config or {})

    # 回填缺失的工作流
    for name in list(workflow_graph.keys()):
        workflow = workflow_graph[name]
        # 获取工作流依赖关系
        deps = [
            d.replace("workflow:", "")
            for d in workflow.workflow.dependencies
            if d.startswith("workflow:")
        ]
        for dependency in deps:
            # 如果依赖关系不在工作流图中，则创建新的工作流对象
            if dependency not in workflow_graph:
                reference = {"name": dependency, **workflow.config}
                workflow_graph[dependency] = WorkflowToRun(
                    workflow=create_workflow(
                        dependency,
                        config=reference,
                        additional_verbs=additional_verbs,
                        additional_workflows=additional_workflows,
                        memory_profile=memory_profile,
                    ),
                    config=reference,
                )

    # 运行工作流，按依赖关系顺序排列
    def filter_wf_dependencies(name: str) -> list[str]:
        """
        过滤工作流依赖关系

        Args:
            - name - 工作流名称
        Returns:
            - 依赖关系列表
        """
        externals = [
            e.replace("workflow:", "")
            for e in workflow_graph[name].workflow.dependencies
        ]
        return [e for e in externals if e in workflow_graph]

    # 创建任务图，存储工作流名称到依赖关系列表的映射
    task_graph = {name: filter_wf_dependencies(name) for name in workflow_graph}
    # 运行工作流，按依赖关系顺序排列
    workflow_run_order = topological_sort(task_graph)
    workflows = [workflow_graph[name] for name in workflow_run_order]
    log.info("Workflow Run Order: %s", workflow_run_order)
    return LoadWorkflowResult(workflows=workflows, dependencies=task_graph)


def create_workflow(
    name: str,
    steps: list[PipelineWorkflowStep] | None = None,
    config: PipelineWorkflowConfig | None = None,
    additional_verbs: VerbDefinitions | None = None,
    additional_workflows: WorkflowDefinitions | None = None,
    memory_profile: bool = False,
) -> Workflow:
    """
    根据给定的配置创建一个工作流。

    Args:
        name (str): 工作流名称
        steps (list[PipelineWorkflowStep] | None): 工作流步骤（可选）
        config (PipelineWorkflowConfig | None): 工作流配置（可选）
        additional_verbs (VerbDefinitions | None): 额外的动词定义（可选）
        additional_workflows (WorkflowDefinitions | None): 额外的工作流定义（可选）
        memory_profile (bool): 是否启用内存分析（默认为 False）

    Returns:
        Workflow: 创建的工作流对象
    """
    # 合并默认工作流和额外工作流定义
    additional_workflows = {
        **_default_workflows,
        **(additional_workflows or {}),
    }

    # 获取工作流步骤，如果没有提供则根据名称和配置获取
    steps = steps or _get_steps_for_workflow(name, config, additional_workflows)
    print(f"{steps=}")  # 输出获取的步骤

    # 移除禁用的步骤
    steps = _remove_disabled_steps(steps)
    print(f"{steps=}")  # 输出移除禁用步骤后的步骤

    # 创建工作流对象
    return Workflow(
        verbs=additional_verbs or {},  # 使用额外的动词定义或空字典
        schema={
            "name": name,  # 工作流名称
            "steps": steps,  # 工作流步骤
        },
        validate=False,  # 不进行验证
        memory_profile=memory_profile,  # 是否启用内存分析
    )


def _get_steps_for_workflow(
    name: str | None,
    config: PipelineWorkflowConfig | None,
    workflows: dict[str, Callable] | None,
) -> list[PipelineWorkflowStep]:
    """Get the steps for the given workflow config."""
    if config is not None and "steps" in config:
        return config["steps"]

    if workflows is None:
        raise NoWorkflowsDefinedError

    if name is None:
        raise UndefinedWorkflowError

    if name not in workflows:
        raise UnknownWorkflowError(name)

    return workflows[name](config or {})


def _remove_disabled_steps(
    steps: list[PipelineWorkflowStep],
) -> list[PipelineWorkflowStep]:
    return [step for step in steps if step.get("enabled", True)]
