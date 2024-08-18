# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Different methods to run the pipeline."""

import gc
import json
import logging
import time
import traceback
from collections.abc import AsyncIterable
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from string import Template
from typing import cast

import pandas as pd
from datashaper import (
    DEFAULT_INPUT_NAME,
    MemoryProfile,
    Workflow,
    WorkflowCallbacks,
    WorkflowCallbacksManager,
    WorkflowRunResult,
)

from .cache import InMemoryCache, PipelineCache, load_cache
from .config import (
    PipelineBlobCacheConfig,
    PipelineBlobReportingConfig,
    PipelineBlobStorageConfig,
    PipelineCacheConfigTypes,
    PipelineConfig,
    PipelineFileCacheConfig,
    PipelineFileReportingConfig,
    PipelineFileStorageConfig,
    PipelineInputConfigTypes,
    PipelineMemoryCacheConfig,
    PipelineReportingConfigTypes,
    PipelineStorageConfigTypes,
    PipelineWorkflowReference,
    PipelineWorkflowStep,
)
from .context import PipelineRunContext, PipelineRunStats
from .emit import TableEmitterType, create_table_emitters
from .input import load_input
from .load_pipeline_config import load_pipeline_config
from .progress import NullProgressReporter, ProgressReporter
from .reporting import (
    ConsoleWorkflowCallbacks,
    ProgressWorkflowCallbacks,
    load_pipeline_reporter,
)
from .storage import MemoryPipelineStorage, PipelineStorage, load_storage
from .typing import PipelineRunResult
# Register all verbs
from .verbs import *  # noqa
from .workflows import (
    VerbDefinitions,
    WorkflowDefinitions,
    create_workflow,
    load_workflows,
)

log = logging.getLogger(__name__)


async def run_pipeline_with_config(
    config_or_path: PipelineConfig | str,
    workflows: list[PipelineWorkflowReference] | None = None,
    dataset: pd.DataFrame | None = None,
    storage: PipelineStorage | None = None,
    cache: PipelineCache | None = None,
    callbacks: WorkflowCallbacks | None = None,
    progress_reporter: ProgressReporter | None = None,
    input_post_process_steps: list[PipelineWorkflowStep] | None = None,
    additional_verbs: VerbDefinitions | None = None,
    additional_workflows: WorkflowDefinitions | None = None,
    emit: list[TableEmitterType] | None = None,
    memory_profile: bool = False,
    run_id: str | None = None,
    is_resume_run: bool = False,
    **_kwargs: dict,
) -> AsyncIterable[PipelineRunResult]:
    """
    使用给定的配置运行管道。

    Args:
        - config_or_path: 要运行的管道配置或路径。
        - workflows: 要运行的工作流（覆盖配置）。
        - dataset: 要运行管道的数据集（覆盖配置）。
        - storage: 要使用的存储（覆盖配置）。
        - cache: 要使用的缓存（覆盖配置）。
        - reporter: 要使用的报告器（覆盖配置）。
        - input_post_process_steps: 要在输入数据上运行的后处理步骤（覆盖配置）。
        - additional_verbs: 自定义动词。
        - additional_workflows: 自定义工作流。
        - emit: 要使用的表格发射器。
        - memory_profile: 是否启用内存分析。
        - run_id: 要启动或恢复的运行ID。
    """

    # 如果配置是字符串，则记录日志
    if isinstance(config_or_path, str):
        log.info("使用配置 %s 运行管道", config_or_path)
    else:
        log.info("运行管道")

    # 生成运行ID
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S")

    # 加载配置
    config = load_pipeline_config(config_or_path)

    # 应用替换
    config = _apply_substitutions(config, run_id)

    # 获取根目录
    root_dir = config.root_dir

    # 创建存储
    def _create_storage(config: PipelineStorageConfigTypes | None) -> PipelineStorage:
        """
        创建存储。

        Args:
            - config: 存储配置。

        Returns:
            - PipelineStorage: 存储实例。
        """
        return load_storage(
            config
            or PipelineFileStorageConfig(base_dir=str(Path(root_dir or "") / "output"))
        )

    # 创建缓存
    def _create_cache(config: PipelineCacheConfigTypes | None) -> PipelineCache:
        """
        创建缓存。

        Args:
            - config: 缓存配置。

        Returns:
            - PipelineCache: 缓存实例。
        """
        return load_cache(config or PipelineMemoryCacheConfig(), root_dir=root_dir)

    # 创建报告器
    def _create_reporter(
        config: PipelineReportingConfigTypes | None,
    ) -> WorkflowCallbacks | None:
        """
        创建报告器。

        Args:
            - config: 报告器配置。

        Returns:
            - WorkflowCallbacks: 报告器实例。
        """
        return load_pipeline_reporter(config, root_dir) if config else None

    # 创建输入数据
    async def _create_input(
        config: PipelineInputConfigTypes | None,
    ) -> pd.DataFrame | None:
        """
        创建输入数据。

        Args:
            - config: 输入数据配置。

        Returns:
            - pd.DataFrame: 输入数据。
        """
        if config is None:
            return None

        return await load_input(config, progress_reporter, root_dir)

    # 创建后处理步骤
    def _create_postprocess_steps(
        config: PipelineInputConfigTypes | None,
    ) -> list[PipelineWorkflowStep] | None:
        """
        创建后处理步骤。

        Args:
            - config: 后处理步骤配置。

        Returns:
            - list[PipelineWorkflowStep]: 后处理步骤。
        """
        return config.post_process if config is not None else None

    # 设置报告器
    progress_reporter = progress_reporter or NullProgressReporter()

    # 设置存储
    storage = storage or _create_storage(config.storage)

    # 打印存储信息
    print(f"{storage=}")

    # 设置缓存
    cache = cache or _create_cache(config.cache)

    # 设置报告器
    callbacks = callbacks or _create_reporter(config.reporting)
    # 设置输入数据
    dataset = dataset if dataset is not None else await _create_input(config.input)

    # 设置后处理步骤
    post_process_steps = input_post_process_steps or _create_postprocess_steps(
        config.input
    )

    # 设置工作流
    workflows = workflows or config.workflows

    # 检查输入数据是否为空
    if dataset is None:
        msg = "没有提供输入数据!"
        raise ValueError(msg)

    # 运行管道
    async for table in run_pipeline(
        workflows=workflows,
        dataset=dataset,
        storage=storage,
        cache=cache,
        callbacks=callbacks,
        input_post_process_steps=post_process_steps,
        memory_profile=memory_profile,
        additional_verbs=additional_verbs,
        additional_workflows=additional_workflows,
        progress_reporter=progress_reporter,
        emit=emit,
        is_resume_run=is_resume_run,
    ):
        yield table


async def run_pipeline(
    workflows: list[PipelineWorkflowReference],
    dataset: pd.DataFrame,
    storage: PipelineStorage | None = None,
    cache: PipelineCache | None = None,
    callbacks: WorkflowCallbacks | None = None,
    progress_reporter: ProgressReporter | None = None,
    input_post_process_steps: list[PipelineWorkflowStep] | None = None,
    additional_verbs: VerbDefinitions | None = None,
    additional_workflows: WorkflowDefinitions | None = None,
    emit: list[TableEmitterType] | None = None,
    memory_profile: bool = False,
    is_resume_run: bool = False,
    **_kwargs: dict,
) -> AsyncIterable[PipelineRunResult]:
    """
    运行管道。

    Args:
        - workflows: 要运行的工作流
        - dataset: 要运行管道的数据集，必须包含以下列：
            - id: 文档ID
            - text: 文档文本
            - title: 文档标题
        - storage: 存储器
        - cache: 缓存器
        - reporter: 报告器
        - input_post_process_steps: 输入数据后处理步骤
        - additional_verbs: 自定义动词
        - additional_workflows: 自定义工作流
        - debug: 是否运行调试模式
    Returns:
        - output: 工作流结果异步迭代器
    """
    # 开始时间
    start_time = time.time()
    # 统计信息
    stats = PipelineRunStats()
    # 存储器
    storage = storage or MemoryPipelineStorage()
    # 缓存器
    cache = cache or InMemoryCache()
    # 报告器
    progress_reporter = progress_reporter or NullProgressReporter()
    # 回调函数
    callbacks = callbacks or ConsoleWorkflowCallbacks()
    # 创建回调链
    callbacks = _create_callback_chain(callbacks, progress_reporter)
    # 发射器
    emit = emit or [TableEmitterType.Parquet]
    # 创建表发射器
    emitters = create_table_emitters(
        emit,
        storage,
        lambda e, s, d: cast(WorkflowCallbacks, callbacks).on_error(
            "Error emitting table", e, s, d
        ),
    )
    # 加载工作流
    loaded_workflows = load_workflows(
        workflows,
        additional_verbs=additional_verbs,
        additional_workflows=additional_workflows,
        memory_profile=memory_profile,
    )
    # 工作流列表
    workflows_to_run = loaded_workflows.workflows
    # 工作流依赖关系
    workflow_dependencies = loaded_workflows.dependencies

    # 打印存储器信息
    print(f"{storage=}")
    # 创建运行上下文
    context = _create_run_context(storage, cache, stats)

    # 如果没有发射器，打印警告信息
    if len(emitters) == 0:
        log.info(
            "No emitters provided. No table outputs will be generated. This is probably not correct."
        )

    # 定义统计信息保存函数
    async def dump_stats() -> None:
        await storage.set(
            "stats.json", json.dumps(asdict(stats), indent=4, ensure_ascii=False)
        )

    # 定义从存储器加载表函数
    async def load_table_from_storage(name: str) -> pd.DataFrame:
        if not await storage.has(name):
            msg = f"Could not find {name} in storage!"
            raise ValueError(msg)
        try:
            log.info("read table from storage: %s", name)
            return pd.read_parquet(BytesIO(await storage.get(name, as_bytes=True)))
        except Exception:
            log.exception("error loading table from storage: %s", name)
            raise

    # 定义注入工作流数据依赖函数
    async def inject_workflow_data_dependencies(workflow: Workflow) -> None:
        workflow.add_table(DEFAULT_INPUT_NAME, dataset)
        deps = workflow_dependencies[workflow.name]
        print("dependencies for %s: %s", workflow.name, deps)
        log.info("dependencies for %s: %s", workflow.name, deps)
        for id in deps:
            workflow_id = f"workflow:{id}"
            table = await load_table_from_storage(f"{id}.parquet")
            workflow.add_table(workflow_id, table)

    # 定义写入工作流统计信息函数
    async def write_workflow_stats(
        workflow: Workflow,
        workflow_result: WorkflowRunResult,
        workflow_start_time: float,
    ) -> None:
        for vt in workflow_result.verb_timings:
            stats.workflows[workflow.name][f"{vt.index}_{vt.verb}"] = vt.timing

        workflow_end_time = time.time()
        stats.workflows[workflow.name]["overall"] = (
            workflow_end_time - workflow_start_time
        )
        stats.total_runtime = time.time() - start_time
        await dump_stats()

        if workflow_result.memory_profile is not None:
            await _save_profiler_stats(
                storage, workflow.name, workflow_result.memory_profile
            )

        log.debug(
            "first row of %s => %s", workflow_name, workflow.output().iloc[0].to_json()
        )

    # 定义发射工作流输出函数
    async def emit_workflow_output(workflow: Workflow) -> pd.DataFrame:
        output = cast(pd.DataFrame, workflow.output())
        for emitter in emitters:
            await emitter.emit(workflow.name, output)
        return output

    # 运行输入数据后处理步骤
    dataset = await _run_post_process_steps(
        input_post_process_steps, dataset, context, callbacks
    )

    # 验证数据集
    _validate_dataset(dataset)

    log.info("Final # of rows loaded: %s", len(dataset))
    stats.num_documents = len(dataset)
    last_workflow = "input"

    try:
        await dump_stats()

        # 运行工作流
        for workflow_to_run in workflows_to_run:
            # 尝试清除中间数据帧
            gc.collect()

            workflow = workflow_to_run.workflow
            workflow_name: str = workflow.name
            last_workflow = workflow_name

            print("Running workflow: %s...", workflow_name)
            log.info("Running workflow: %s...", workflow_name)

            if is_resume_run and await storage.has(
                f"{workflow_to_run.workflow.name}.parquet"
            ):
                log.info("Skipping %s because it already exists", workflow_name)
                continue

            stats.workflows[workflow_name] = {"overall": 0.0}
            await inject_workflow_data_dependencies(workflow)

            workflow_start_time = time.time()
            result = await workflow.run(context, callbacks)
            await write_workflow_stats(workflow, result, workflow_start_time)

            # 保存工作流输出
            output = await emit_workflow_output(workflow)
            yield PipelineRunResult(workflow_name, output, None)
            output = None
            workflow.dispose()
            workflow = None

        stats.total_runtime = time.time() - start_time
        await dump_stats()
    except Exception as e:
        log.exception("error running workflow %s", last_workflow)
        cast(WorkflowCallbacks, callbacks).on_error(
            "Error running pipeline!", e, traceback.format_exc()
        )
        yield PipelineRunResult(last_workflow, None, [e])


def _create_callback_chain(
    callbacks: WorkflowCallbacks | None, progress: ProgressReporter | None
) -> WorkflowCallbacks:
    """
    创建回调链管理器。

    Args:
        callbacks (WorkflowCallbacks | None): 回调函数。
        progress (ProgressReporter | None): 进度报告器。

    Returns:
        WorkflowCallbacks: 回调链管理器。
    """
    # 创建回调链管理器实例
    manager = WorkflowCallbacksManager()

    # 注册回调函数
    if callbacks is not None:
        manager.register(callbacks)

    # 注册进度报告器
    if progress is not None:
        manager.register(ProgressWorkflowCallbacks(progress))

    # 返回回调链管理器
    return manager


async def _save_profiler_stats(
    storage: PipelineStorage, workflow_name: str, profile: MemoryProfile
):
    """
    保存性能分析数据到存储。

    Args:
        storage (PipelineStorage): 存储实例。
        workflow_name (str): 工作流名称。
        profile (MemoryProfile): 性能分析数据。
    """
    # 保存峰值统计数据
    await storage.set(
        f"{workflow_name}_profiling.peak_stats.csv",
        profile.peak_stats.to_csv(index=True),
    )

    # 保存快照统计数据
    await storage.set(
        f"{workflow_name}_profiling.snapshot_stats.csv",
        profile.snapshot_stats.to_csv(index=True),
    )

    # 保存时间统计数据
    await storage.set(
        f"{workflow_name}_profiling.time_stats.csv",
        profile.time_stats.to_csv(index=True),
    )

    # 保存详细视图数据
    await storage.set(
        f"{workflow_name}_profiling.detailed_view.csv",
        profile.detailed_view.to_csv(index=True),
    )


async def _run_post_process_steps(
    post_process: list[PipelineWorkflowStep] | None,
    dataset: pd.DataFrame,
    context: PipelineRunContext,
    callbacks: WorkflowCallbacks,
) -> pd.DataFrame:
    """
    运行后处理步骤

    Args:
        - post_process: 需要运行的后处理步骤
        - dataset: 需要处理的数据集
        - context: 流水线运行上下文
    Returns:
        - output: 运行后处理步骤后的数据集
    """
    # 检查是否有后处理步骤需要运行
    if post_process is not None and len(post_process) > 0:
        # 创建一个新的工作流来运行后处理步骤
        input_workflow = create_workflow(
            "Input Post Process",
            post_process,
        )
        # 打印数据集信息
        print(f"{dataset=}")
        # 将数据集添加到工作流中
        input_workflow.add_table(DEFAULT_INPUT_NAME, dataset)
        # 运行工作流
        await input_workflow.run(
            context=context,
            callbacks=callbacks,
        )
        # 获取工作流输出的数据集
        dataset = cast(pd.DataFrame, input_workflow.output())
    # 返回处理后的数据集
    return dataset


def _validate_dataset(dataset: pd.DataFrame):
    """
    验证数据集是否符合管道要求。

    Args:
        - dataset: 需要验证的数据集

    Raises:
        TypeError: 如果数据集不是 pandas DataFrame 类型
    """
    # 检查数据集是否是 pandas DataFrame 类型
    if not isinstance(dataset, pd.DataFrame):
        # 如果不是，抛出 TypeError
        msg = "数据集必须是 pandas DataFrame 类型！"
        raise TypeError(msg)


def _apply_substitutions(config: PipelineConfig, run_id: str) -> PipelineConfig:
    """
    应用配置替换

    将 run_id 替换到配置中的 storage、cache 和 reporting 的 base_dir 中

    Args:
        config (PipelineConfig): 配置对象
        run_id (str): 运行 ID

    Returns:
        PipelineConfig: 替换后的配置对象
    """
    # 定义替换字典
    substitutions = {"timestamp": run_id}

    # 替换 storage 的 base_dir
    if (
        isinstance(
            config.storage, PipelineFileStorageConfig | PipelineBlobStorageConfig
        )
        and config.storage.base_dir
    ):
        # 如果 storage 的 base_dir 存在，则替换 timestamp
        config.storage.base_dir = Template(config.storage.base_dir).substitute(
            substitutions
        )

    # 替换 cache 的 base_dir
    if (
        isinstance(config.cache, PipelineFileCacheConfig | PipelineBlobCacheConfig)
        and config.cache.base_dir
    ):
        # 如果 cache 的 base_dir 存在，则替换 timestamp
        config.cache.base_dir = Template(config.cache.base_dir).substitute(
            substitutions
        )

    # 替换 reporting 的 base_dir
    if (
        isinstance(
            config.reporting, PipelineFileReportingConfig | PipelineBlobReportingConfig
        )
        and config.reporting.base_dir
    ):
        # 如果 reporting 的 base_dir 存在，则替换 timestamp
        config.reporting.base_dir = Template(config.reporting.base_dir).substitute(
            substitutions
        )

    # 返回替换后的配置对象
    return config


def _create_run_context(
    storage: PipelineStorage,  # 流水线存储对象
    cache: PipelineCache,  # 流水线缓存对象
    stats: PipelineRunStats,  # 流水线运行统计对象
) -> PipelineRunContext:
    """
    创建流水线运行上下文。

    Args:
        storage (PipelineStorage): 流水线存储对象。
        cache (PipelineCache): 流水线缓存对象。
        stats (PipelineRunStats): 流水线运行统计对象。

    Returns:
        PipelineRunContext: 流水线运行上下文对象。
    """
    # 创建流水线运行上下文对象
    run_context = PipelineRunContext(
        stats=stats,  # 设置运行统计对象
        cache=cache,  # 设置缓存对象
        storage=storage,  # 设置存储对象
    )
    return run_context
