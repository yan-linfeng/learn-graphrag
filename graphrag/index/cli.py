# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Main definition."""

import asyncio
import json
import logging
import platform
import sys
import time
import warnings
from pathlib import Path

from graphrag.config import (
    create_graphrag_config,
)
from graphrag.index import PipelineConfig, create_pipeline_config
from graphrag.index.cache import NoopPipelineCache
from graphrag.index.progress import (
    NullProgressReporter,
    PrintProgressReporter,
    ProgressReporter,
)
from graphrag.index.progress.rich import RichProgressReporter
from graphrag.index.run import run_pipeline_with_config
from .emit import TableEmitterType
from .graph.extractors.claims.prompts import CLAIM_EXTRACTION_PROMPT
from .graph.extractors.community_reports.prompts import COMMUNITY_REPORT_PROMPT
from .graph.extractors.graph.prompts import GRAPH_EXTRACTION_PROMPT
from .graph.extractors.summarize.prompts import SUMMARIZE_PROMPT
from .init_content import INIT_DOTENV, INIT_YAML

# Ignore warnings from numba
warnings.filterwarnings("ignore", message=".*NumbaDeprecationWarning.*")

log = logging.getLogger(__name__)


# Redact any sensitive configuration
def redact(input: dict) -> str:
    """
    对敏感配置进行屏蔽，并返回屏蔽后的JSON字符串

    Args:
        input (dict): 输入的字典

    Returns:
        str: 屏蔽后的JSON字符串
    """

    # 屏蔽敏感配置
    def redact_dict(input: dict) -> dict:
        """
        递归屏蔽字典中的敏感配置

        Args:
            input (dict): 输入的字典

        Returns:
            dict: 屏蔽后的字典
        """
        if not isinstance(input, dict):
            return input

        result = {}
        for key, value in input.items():
            if key in {
                "api_key",
                "connection_string",
                "container_name",
                "organization",
                "organization",
                "compartment_id",
            }:
                if value is not None:
                    result[key] = f"REDACTED, length {len(value)}"
            elif isinstance(value, dict):
                result[key] = redact_dict(value)
            elif isinstance(value, list):
                result[key] = [redact_dict(i) for i in value]
            else:
                result[key] = value
        return result

    # 对输入进行屏蔽
    redacted_dict = redact_dict(input)

    # 将屏蔽后的字典转换为带缩进的JSON字符串
    return json.dumps(redacted_dict, indent=4)


def index_cli(
    root: str,
    init: bool,
    verbose: bool,
    resume: str | None,
    memprofile: bool,
    nocache: bool,
    reporter: str | None,
    config: str | None,
    emit: str | None,
    dryrun: bool,
    overlay_defaults: bool,
    cli: bool = False,
):
    """
    运行管道并使用给定的配置。

    :param root: 根目录
    :param init: 是否初始化项目
    :param verbose: 是否输出详细日志
    :param resume: 恢复运行的ID
    :param memprofile: 是否输出内存使用情况
    :param nocache: 是否禁用缓存
    :param reporter: 报告器类型
    :param config: 配置文件路径
    :param emit: 输出类型
    :param dryrun: 是否仅输出日志而不实际运行
    :param overlay_defaults: 是否覆盖默认配置
    :param cli: 是否为CLI模式
    """
    # 获取运行ID
    run_id = resume or time.strftime("%Y%m%d-%H%M%S")
    # 启用日志输出
    _enable_logging(root, run_id, verbose)
    # 获取报告器实例
    progress_reporter = _get_progress_reporter(reporter)

    # 如果初始化项目，则退出
    if init:
        _initialize_project_at(root, progress_reporter)
        sys.exit(0)

    # 如果覆盖默认配置，则创建默认配置
    if overlay_defaults:
        pipeline_config: str | PipelineConfig = _create_default_config(
            root, config, verbose, dryrun or False, progress_reporter
        )
    else:
        # 否则使用配置文件或创建默认配置
        pipeline_config: str | PipelineConfig = config or _create_default_config(
            root, None, verbose, dryrun or False, progress_reporter
        )

    # 创建缓存实例
    cache = NoopPipelineCache() if nocache else None
    # 解析输出类型
    pipeline_emit = emit.split(",") if emit else None
    # 初始化错误标志
    encountered_errors = False

    def _run_workflow_async() -> None:
        """
        异步运行工作流。
        """
        import signal

        def handle_signal(signum, _):
            """
            处理信号。

            :param signum: 信号号
            :param _: 信号参数
            """
            # 输出信号信息
            progress_reporter.info(f"Received signal {signum}, exiting...")
            # 释放报告器资源
            progress_reporter.dispose()
            # 取消所有任务
            for task in asyncio.all_tasks():
                task.cancel()
            # 输出任务取消信息
            progress_reporter.info("All tasks cancelled. Exiting...")

        # 注册信号处理器
        signal.signal(signal.SIGINT, handle_signal)

        # 如果不是Windows平台，则注册SIGHUP信号处理器
        if sys.platform != "win32":
            signal.signal(signal.SIGHUP, handle_signal)

        async def execute():
            """
            执行工作流。
            """
            nonlocal encountered_errors
            # 运行管道并获取输出
            async for output in run_pipeline_with_config(
                pipeline_config,
                run_id=run_id,
                memory_profile=memprofile,
                cache=cache,
                progress_reporter=progress_reporter,
                emit=(
                    [TableEmitterType(e) for e in pipeline_emit]
                    if pipeline_emit
                    else None
                ),
                is_resume_run=bool(resume),
            ):
                # 如果输出包含错误，则设置错误标志
                if output.errors and len(output.errors) > 0:
                    encountered_errors = True
                    # 输出错误信息
                    progress_reporter.error(output.workflow)
                else:
                    # 输出成功信息
                    progress_reporter.success(output.workflow)

                # 输出结果信息
                progress_reporter.info(str(output.result))

        # 如果是Windows平台，则使用nest_asyncio运行异步任务
        if platform.system() == "Windows":
            import nest_asyncio  # type: ignore

            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            loop.run_until_complete(execute())
        # 如果是Python 3.11或以上版本，则使用uvloop运行异步任务
        elif sys.version_info >= (3, 11):
            import uvloop  # type: ignore

            with asyncio.Runner(
                loop_factory=uvloop.new_event_loop) as runner:  # type: ignore
                runner.run(execute())
        # 否则使用uvloop运行异步任务
        else:
            import uvloop  # type: ignore

            uvloop.install()
            asyncio.run(execute())

    # 运行异步工作流
    _run_workflow_async()
    # 停止报告器
    progress_reporter.stop()
    # 如果发生错误，则输出错误信息
    if encountered_errors:
        progress_reporter.error(
            "Errors occurred during the pipeline run, see logs for more details."
        )
    else:
        # 否则输出成功信息
        progress_reporter.success("All workflows completed successfully.")

    # 如果是CLI模式，则退出
    if cli:
        sys.exit(1 if encountered_errors else 0)


def _initialize_project_at(path: str, reporter: ProgressReporter) -> None:
    """Initialize the project at the given path."""
    reporter.info(f"Initializing project at {path}")
    root = Path(path)
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)

    settings_yaml = root / "settings.yaml"
    if settings_yaml.exists():
        msg = f"Project already initialized at {root}"
        raise ValueError(msg)

    dotenv = root / ".env"
    if not dotenv.exists():
        with settings_yaml.open("wb") as file:
            file.write(INIT_YAML.encode(encoding="utf-8", errors="strict"))

    with dotenv.open("wb") as file:
        file.write(INIT_DOTENV.encode(encoding="utf-8", errors="strict"))

    prompts_dir = root / "prompts"
    if not prompts_dir.exists():
        prompts_dir.mkdir(parents=True, exist_ok=True)

    entity_extraction = prompts_dir / "entity_extraction.txt"
    if not entity_extraction.exists():
        with entity_extraction.open("wb") as file:
            file.write(
                GRAPH_EXTRACTION_PROMPT.encode(encoding="utf-8", errors="strict")
            )

    summarize_descriptions = prompts_dir / "summarize_descriptions.txt"
    if not summarize_descriptions.exists():
        with summarize_descriptions.open("wb") as file:
            file.write(SUMMARIZE_PROMPT.encode(encoding="utf-8", errors="strict"))

    claim_extraction = prompts_dir / "claim_extraction.txt"
    if not claim_extraction.exists():
        with claim_extraction.open("wb") as file:
            file.write(
                CLAIM_EXTRACTION_PROMPT.encode(encoding="utf-8", errors="strict")
            )

    community_report = prompts_dir / "community_report.txt"
    if not community_report.exists():
        with community_report.open("wb") as file:
            file.write(
                COMMUNITY_REPORT_PROMPT.encode(encoding="utf-8", errors="strict")
            )


def _create_default_config(
    root: str,
    config: str | None,
    verbose: bool,
    dryrun: bool,
    reporter: ProgressReporter,
) -> PipelineConfig:
    """
    如果提供了配置文件，则在其基础上覆盖默认值；如果没有提供配置文件，则创建一个默认配置。

    :param root: 根目录
    :param config: 配置文件路径（可选）
    :param verbose: 是否输出详细信息
    :param dryrun: 是否仅进行演练（不实际执行）
    :param reporter: 进度报告器
    :return: PipelineConfig 对象
    """

    # 检查配置文件是否存在
    if config and not Path(config).exists():
        msg = f"配置文件 {config} 不存在"
        raise ValueError(msg)

    # 检查根目录是否存在
    if not Path(root).exists():
        msg = f"根目录 {root} 不存在"
        raise ValueError(msg)

    # 读取配置参数
    parameters = _read_config_parameters(root, config, reporter)
    if verbose or dryrun:
        reporter.info(f"parameters: {parameters}")

    # 输出默认配置信息
    log.info(
        "使用默认配置: %s",
        redact(parameters.model_dump()),
    )

    # 如果需要输出详细信息或仅进行演练，则输出配置信息
    if verbose or dryrun:
        reporter.info(f"使用默认配置: {redact(parameters.model_dump())}")

    # 创建管道配置
    result = create_pipeline_config(parameters, verbose)

    # 如果需要输出详细信息或仅进行演练，则输出最终配置信息
    if verbose or dryrun:
        reporter.info(f"最终配置: {redact(result.model_dump())}")

    # 如果仅进行演练，则退出
    if dryrun:
        reporter.info("演练完成，退出...")
        sys.exit(0)

    # 返回管道配置
    return result


def _read_config_parameters(root: str, config: str | None, reporter: ProgressReporter):
    """
    读取配置参数

    :param root: 根目录
    :param config: 配置文件路径（可选）
    :param reporter: 进度报告器
    :return: 配置参数
    """
    # 获取根目录路径
    _root = Path(root)

    # 获取配置文件路径（优先使用 config 参数，如果不存在则使用默认路径）
    settings_yaml = (
        Path(config)
        if config and Path(config).suffix in [".yaml", ".yml"]
        else _root / "settings.yaml"
    )
    # 如果 settings.yaml 不存在，则尝试使用 settings.yml
    if not settings_yaml.exists():
        settings_yaml = _root / "settings.yml"

    # 获取 JSON 配置文件路径（优先使用 config 参数，如果不存在则使用默认路径）
    settings_json = (
        Path(config)
        if config and Path(config).suffix == ".json"
        else _root / "settings.json"
    )

    # 如果 YAML 配置文件存在，则读取配置
    if settings_yaml.exists():
        # 报告读取配置文件
        reporter.success(f"Reading settings from {settings_yaml}")
        # 读取 YAML 配置文件
        with settings_yaml.open("rb") as file:
            import yaml

            # 解析 YAML 配置文件
            data = yaml.safe_load(file.read().decode(encoding="utf-8", errors="strict"))
            # 创建配置参数
            return create_graphrag_config(data, root)

    # 如果 JSON 配置文件存在，则读取配置
    if settings_json.exists():
        # 报告读取配置文件
        reporter.success(f"Reading settings from {settings_json}")
        # 读取 JSON 配置文件
        with settings_json.open("rb") as file:
            import json

            # 解析 JSON 配置文件
            data = json.loads(file.read().decode(encoding="utf-8", errors="strict"))
            # 创建配置参数
            return create_graphrag_config(data, root)

    # 如果没有配置文件，则从环境变量中读取配置
    reporter.success("Reading settings from environment variables")
    # 创建配置参数
    return create_graphrag_config(root_dir=root)


def _get_progress_reporter(reporter_type: str | None) -> ProgressReporter:
    if reporter_type is None or reporter_type == "rich":
        return RichProgressReporter("GraphRAG Indexer ")
    if reporter_type == "print":
        return PrintProgressReporter("GraphRAG Indexer ")
    if reporter_type == "none":
        return NullProgressReporter()

    msg = f"Invalid progress reporter type: {reporter_type}"
    raise ValueError(msg)


def _enable_logging(root_dir: str, run_id: str, verbose: bool) -> None:
    """
    启用日志记录功能

    :param root_dir: 根目录
    :param run_id: 运行ID
    :param verbose: 是否启用详细日志
    :return: None
    """
    # 指定日志文件路径
    logging_file = (
        Path(root_dir) / "output" / run_id / "reports" / "indexing-engine.log"
    )

    # 创建日志文件所在目录（如果不存在）
    logging_file.parent.mkdir(parents=True, exist_ok=True)

    # 创建日志文件（如果不存在）
    logging_file.touch(exist_ok=True)

    # 配置基本日志记录设置
    logging.basicConfig(
        # 日志文件路径
        filename=str(logging_file),
        # 日志文件模式（追加模式）
        filemode="a",
        # 日志格式
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        # 日期格式
        datefmt="%H:%M:%S",
        # 日志级别（根据verbose参数决定）
        level=logging.DEBUG if verbose else logging.INFO,
    )
