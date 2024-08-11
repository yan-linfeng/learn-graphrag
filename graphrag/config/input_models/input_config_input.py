# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""配置参数的默认设置。"""

from typing_extensions import NotRequired, TypedDict

from graphrag.config.enums import InputFileType, InputType


class InputConfigInput(TypedDict):
    """
    输入配置的默认配置节。

    该类定义了输入配置的参数，包括类型、文件类型、基本目录、连接字符串、容器名称、文件编码、文件模式、源列、时间戳列、时间戳格式、文本列、标题列、文档属性列和存储账户 Blob URL。
    """

    # 输入类型（可选）
    type: NotRequired[InputType | str | None]  # 输入类型，可以是 InputType 或字符串

    # 文件类型（可选）
    file_type: NotRequired[InputFileType | str | None]  # 文件类型，可以是 InputFileType 或字符串

    # 基本目录（可选）
    base_dir: NotRequired[str | None]  # 基本目录，可以是字符串

    # 连接字符串（可选）
    connection_string: NotRequired[str | None]  # 连接字符串，可以是字符串

    # 容器名称（可选）
    container_name: NotRequired[str | None]  # 容器名称，可以是字符串

    # 文件编码（可选）
    file_encoding: NotRequired[str | None]  # 文件编码，可以是字符串

    # 文件模式（可选）
    file_pattern: NotRequired[str | None]  # 文件模式，可以是字符串

    # 源列（可选）
    source_column: NotRequired[str | None]  # 源列，可以是字符串

    # 时间戳列（可选）
    timestamp_column: NotRequired[str | None]  # 时间戳列，可以是字符串

    # 时间戳格式（可选）
    timestamp_format: NotRequired[str | None]  # 时间戳格式，可以是字符串

    # 文本列（可选）
    text_column: NotRequired[str | None]  # 文本列，可以是字符串

    # 标题列（可选）
    title_column: NotRequired[str | None]  # 标题列，可以是字符串

    # 文档属性列（可选）
    document_attribute_columns: NotRequired[list[str] | str | None]  # 文档属性列，可以是列表或字符串

    # 存储账户 Blob URL（可选）
    storage_account_blob_url: NotRequired[str | None]  # 存储账户 Blob URL，可以是字符串
