# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""索引引擎包根目录。"""

import argparse

from .cli import index_cli


def main():
    """
    主函数，负责解析命令行参数并调用index_cli函数。

    :return: None
    """
    # 创建ArgumentParser对象，用于解析命令行参数
    parser = argparse.ArgumentParser()

    # 添加命令行参数：--config，用于指定配置文件
    parser.add_argument(
        "--config",
        help="运行管道时使用的配置yaml文件",
        required=False,
        type=str,
    )

    # 添加命令行参数：-v或--verbose，用于启用详细日志
    parser.add_argument(
        "-v",
        "--verbose",
        help="启用详细日志",
        action="store_true",
    )

    # 添加命令行参数：--memprofile，用于启用内存分析
    parser.add_argument(
        "--memprofile",
        help="启用内存分析",
        action="store_true",
    )

    # 添加命令行参数：--root，用于指定根目录
    parser.add_argument(
        "--root",
        help="如果未指定配置文件，则使用的根目录",
        required=False,
        default="./data",
        type=str,
    )

    # 添加命令行参数：--resume，用于恢复数据运行
    parser.add_argument(
        "--resume",
        help="恢复数据运行",
        required=False,
        default=None,
        type=str,
    )

    # 添加命令行参数：--reporter，用于指定进度报告器
    parser.add_argument(
        "--reporter",
        help="使用的进度报告器，有效值为'rich'、'print'或'none'",
        type=str,
    )

    # 添加命令行参数：--emit，用于指定输出格式
    parser.add_argument(
        "--emit",
        help="输出格式，有效值为'parquet'和'csv'",
        type=str,
    )

    # 添加命令行参数：--dryrun，用于启用管道测试
    parser.add_argument(
        "--dryrun",
        help="启用管道测试",
        action="store_true",
    )

    # 添加命令行参数：--nocache，用于禁用LLM缓存
    parser.add_argument(
        "--nocache",
        help="禁用LLM缓存",
        action="store_true",
    )

    # 添加命令行参数：--init，用于创建初始配置
    parser.add_argument(
        "--init",
        help="创建初始配置",
        action="store_true",
    )

    # 添加命令行参数：--overlay-defaults，用于覆盖默认配置
    parser.add_argument(
        "--overlay-defaults",
        help="覆盖默认配置",
        action="store_true",
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 检查--overlay-defaults参数是否需要--config参数
    if args.overlay_defaults and not args.config:
        parser.error("--overlay-defaults需要--config")

    # 调用index_cli函数
    index_cli(
        root=args.root,
        verbose=args.verbose or False,
        resume=args.resume,
        memprofile=args.memprofile or False,
        nocache=args.nocache or False,
        reporter=args.reporter,
        config=args.config,
        emit=args.emit,
        dryrun=args.dryrun or False,
        init=args.init or False,
        overlay_defaults=args.overlay_defaults or False,
        cli=True,
    )


if __name__ == "__main__":
    main()
