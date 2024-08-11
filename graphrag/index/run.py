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
    """Run a pipeline with the given config.

    Args:
        - config_or_path - The config to run the pipeline with
        - workflows - The workflows to run (this overrides the config)
        - dataset - The dataset to run the pipeline on (this overrides the config)
        - storage - The storage to use for the pipeline (this overrides the config)
        - cache - The cache to use for the pipeline (this overrides the config)
        - reporter - The reporter to use for the pipeline (this overrides the config)
        - input_post_process_steps - The post process steps to run on the input data (this overrides the config)
        - additional_verbs - The custom verbs to use for the pipeline.
        - additional_workflows - The custom workflows to use for the pipeline.
        - emit - The table emitters to use for the pipeline.
        - memory_profile - Whether or not to profile the memory.
        - run_id - The run id to start or resume from.
    """
    if isinstance(config_or_path, str):
        log.info("Running pipeline with config %s", config_or_path)
    else:
        log.info("Running pipeline")

    run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
    config = load_pipeline_config(config_or_path)
    config = _apply_substitutions(config, run_id)
    root_dir = config.root_dir

    def _create_storage(config: PipelineStorageConfigTypes | None) -> PipelineStorage:
        return load_storage(
            config
            or PipelineFileStorageConfig(base_dir=str(Path(root_dir or "") / "output"))
        )

    def _create_cache(config: PipelineCacheConfigTypes | None) -> PipelineCache:
        return load_cache(config or PipelineMemoryCacheConfig(), root_dir=root_dir)

    def _create_reporter(
        config: PipelineReportingConfigTypes | None,
    ) -> WorkflowCallbacks | None:
        return load_pipeline_reporter(config, root_dir) if config else None

    async def _create_input(
        config: PipelineInputConfigTypes | None,
    ) -> pd.DataFrame | None:
        if config is None:
            return None

        return await load_input(config, progress_reporter, root_dir)

    def _create_postprocess_steps(
        config: PipelineInputConfigTypes | None,
    ) -> list[PipelineWorkflowStep] | None:
        return config.post_process if config is not None else None

    progress_reporter = progress_reporter or NullProgressReporter()
    storage = storage or _create_storage(config.storage)
    print(f"{storage=}")
    cache = cache or _create_cache(config.cache)
    callbacks = callbacks or _create_reporter(config.reporting)
    dataset = dataset if dataset is not None else await _create_input(config.input)
    post_process_steps = input_post_process_steps or _create_postprocess_steps(
        config.input
    )
    workflows = workflows or config.workflows

    if dataset is None:
        msg = "No dataset provided!"
        raise ValueError(msg)

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
    """Run the pipeline.

    Args:
        - workflows - The workflows to run
        - dataset - The dataset to run the pipeline on, specifically a dataframe with the following columns at a minimum:
            - id - The id of the document
            - text - The text of the document
            - title - The title of the document
            These must exist after any post process steps are run if there are any!
        - storage - The storage to use for the pipeline
        - cache - The cache to use for the pipeline
        - reporter - The reporter to use for the pipeline
        - input_post_process_steps - The post process steps to run on the input data
        - additional_verbs - The custom verbs to use for the pipeline
        - additional_workflows - The custom workflows to use for the pipeline
        - debug - Whether or not to run in debug mode
    Returns:
        - output - An iterable of workflow results as they complete running, as well as any errors that occur
    """
    start_time = time.time()
    stats = PipelineRunStats()
    storage = storage or MemoryPipelineStorage()
    cache = cache or InMemoryCache()
    progress_reporter = progress_reporter or NullProgressReporter()
    callbacks = callbacks or ConsoleWorkflowCallbacks()
    callbacks = _create_callback_chain(callbacks, progress_reporter)
    emit = emit or [TableEmitterType.Parquet]
    emitters = create_table_emitters(
        emit,
        storage,
        lambda e, s, d: cast(WorkflowCallbacks, callbacks).on_error(
            "Error emitting table", e, s, d
        ),
    )
    loaded_workflows = load_workflows(
        workflows,
        additional_verbs=additional_verbs,
        additional_workflows=additional_workflows,
        memory_profile=memory_profile,
    )
    workflows_to_run = loaded_workflows.workflows
    workflow_dependencies = loaded_workflows.dependencies

    print(f"{storage=}")
    context = _create_run_context(storage, cache, stats)

    if len(emitters) == 0:
        log.info(
            "No emitters provided. No table outputs will be generated. This is probably not correct."
        )

    async def dump_stats() -> None:
        await storage.set(
            "stats.json", json.dumps(asdict(stats), indent=4, ensure_ascii=False)
        )

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

    async def inject_workflow_data_dependencies(workflow: Workflow) -> None:
        workflow.add_table(DEFAULT_INPUT_NAME, dataset)
        deps = workflow_dependencies[workflow.name]
        log.info("dependencies for %s: %s", workflow.name, deps)
        for id in deps:
            workflow_id = f"workflow:{id}"
            table = await load_table_from_storage(f"{id}.parquet")
            workflow.add_table(workflow_id, table)

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

    async def emit_workflow_output(workflow: Workflow) -> pd.DataFrame:
        output = cast(pd.DataFrame, workflow.output())
        for emitter in emitters:
            await emitter.emit(workflow.name, output)
        return output

    dataset = await _run_post_process_steps(
        input_post_process_steps, dataset, context, callbacks
    )

    # Make sure the incoming data is valid
    _validate_dataset(dataset)

    log.info("Final # of rows loaded: %s", len(dataset))
    stats.num_documents = len(dataset)
    last_workflow = "input"

    try:
        await dump_stats()

        for workflow_to_run in workflows_to_run:
            # Try to flush out any intermediate dataframes
            gc.collect()

            workflow = workflow_to_run.workflow
            workflow_name: str = workflow.name
            last_workflow = workflow_name

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

            # Save the output from the workflow
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
    """Create a callbacks manager."""
    manager = WorkflowCallbacksManager()
    if callbacks is not None:
        manager.register(callbacks)
    if progress is not None:
        manager.register(ProgressWorkflowCallbacks(progress))
    return manager


async def _save_profiler_stats(
    storage: PipelineStorage, workflow_name: str, profile: MemoryProfile
):
    """Save the profiler stats to the storage."""
    await storage.set(
        f"{workflow_name}_profiling.peak_stats.csv",
        profile.peak_stats.to_csv(index=True),
    )

    await storage.set(
        f"{workflow_name}_profiling.snapshot_stats.csv",
        profile.snapshot_stats.to_csv(index=True),
    )

    await storage.set(
        f"{workflow_name}_profiling.time_stats.csv",
        profile.time_stats.to_csv(index=True),
    )

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
