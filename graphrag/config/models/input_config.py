# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""输入配置参数设置。"""

from pydantic import BaseModel, Field

import graphrag.config.defaults as defs
from graphrag.config.enums import InputFileType, InputType


class InputConfig(BaseModel):
    """
    输入配置部分的默认配置。

    Attributes:
        type (InputType): 输入类型，默认为 defs.INPUT_TYPE。
        file_type (InputFileType): 输入文件类型，默认为 defs.INPUT_FILE_TYPE。
        base_dir (str): 输入的基本目录路径，默认为 defs.INPUT_BASE_DIR。
        connection_string (str | None): Azure Blob 存储的连接字符串，默认为 None。
        storage_account_blob_url (str | None): Azure Blob 存储的帐户 Blob URL，默认为 None。
        container_name (str | None): Azure Blob 存储的容器名称，默认为 None。
        encoding (str | None): 输入文件的编码格式，默认为 defs.INPUT_FILE_ENCODING。
        file_pattern (str): 输入文件的模式，默认为 defs.INPUT_TEXT_PATTERN。
        file_filter (dict[str, str] | None): 对输入文件的可选过滤器，默认为 None。
        source_column (str | None): 输入的源列，默认为 None。
        timestamp_column (str | None): 输入的时间戳列，默认为 None。
        timestamp_format (str | None): 输入的时间戳格式，默认为 None。
        text_column (str): 输入的文本列，默认为 defs.INPUT_TEXT_COLUMN。
        title_column (str | None): 输入的标题列，默认为 None。
        document_attribute_columns (list[str]): 输入的文档属性列，默认为空列表。
    """

    type: InputType = Field(
        description="要使用的输入类型。", default=defs.INPUT_TYPE
    )
    file_type: InputFileType = Field(
        description="要使用的输入文件类型。", default=defs.INPUT_FILE_TYPE
    )
    base_dir: str = Field(
        description="要使用的输入基本目录。", default=defs.INPUT_BASE_DIR
    )
    connection_string: str | None = Field(
        description="要使用的Azure Blob存储连接字符串。", default=None
    )
    storage_account_blob_url: str | None = Field(
        description="要使用的Azure Blob存储帐户Blob URL。", default=None
    )
    container_name: str | None = Field(
        description="要使用的Azure Blob存储容器名称。", default=None
    )
    encoding: str | None = Field(
        description="要使用的输入文件编码。", default=defs.INPUT_FILE_ENCODING
    )
    file_pattern: str = Field(
        description="要使用的输入文件模式。", default=defs.INPUT_TEXT_PATTERN
    )
    file_filter: dict[str, str] | None = Field(
        description="可选的输入文件过滤器。", default=None
    )
    source_column: str | None = Field(
        description="要使用的输入源列。", default=None
    )
    timestamp_column: str | None = Field(
        description="要使用的输入时间戳列。", default=None
    )
    timestamp_format: str | None = Field(
        description="要使用的输入时间戳格式。", default=None
    )
    text_column: str = Field(
        description="要使用的输入文本列。", default=defs.INPUT_TEXT_COLUMN
    )
    title_column: str | None = Field(
        description="要使用的输入标题列。", default=None
    )
    document_attribute_columns: list[str] = Field(
        description="要使用的输入文档属性列。", default=[]
    )