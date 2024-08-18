# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""配置读取工具类"""

from collections.abc import Callable
from contextlib import contextmanager
from enum import Enum
from typing import Any, TypeVar

from environs import Env

T = TypeVar("T")

# 键值类型定义
KeyValue = str | Enum
EnvKeySet = str | list[str]


def read_key(value: KeyValue) -> str:
    """
    读取键值。

    如果键值是枚举类型，则返回枚举值的字符串表示。
    如果键值是字符串类型，则返回小写字符串。

    Args:
        value: 键值

    Returns:
        str: 读取后的键值
    """
    # 如果键值不是字符串类型，则返回枚举值的字符串表示
    if not isinstance(value, str):
        return value.value.lower()
    # 如果键值是字符串类型，则返回小写字符串
    return value.lower()


class EnvironmentReader:
    """
    配置读取工具类。

    提供环境变量读取和配置管理功能。
    """

    # 环境变量对象
    _env: Env
    # 配置栈
    _config_stack: list[dict]

    def __init__(self, env: Env):
        """
        初始化环境读取器。

        Args:
            env: 环境变量对象
        """
        self._env = env
        self._config_stack = []

    @property
    def env(self):
        """
        获取环境变量对象。

        Returns:
            Env: 环境变量对象
        """
        return self._env

    def _read_env(
        self, env_key: str | list[str], default_value: T, read: Callable[[str, T], T]
    ) -> T | None:
        """
        读取环境变量。

        如果环境变量键是字符串类型，则转换为列表类型。
        按照环境变量键列表顺序读取环境变量，如果读取成功则返回读取值。

        Args:
            env_key: 环境变量键
            default_value: 默认值
            read: 读取函数

        Returns:
            T | None: 读取值或默认值
        """
        # 如果环境变量键是字符串类型，则转换为列表类型
        if isinstance(env_key, str):
            env_key = [env_key]

        # 按照环境变量键列表顺序读取环境变量
        for k in env_key:
            result = read(k.upper(), default_value)
            # 如果读取成功则返回读取值
            if result is not default_value:
                return result

        # 如果读取失败则返回默认值
        return default_value

    def envvar_prefix(self, prefix: KeyValue) -> Env:
        """
        设置环境变量前缀。

        Args:
            prefix (KeyValue): 前缀的值。

        Returns:
            Env: 带有前缀的环境变量对象。
        """
        # 读取前缀的值
        prefix = read_key(prefix)
        # 将前缀转换为大写并添加下划线
        prefix = f"{prefix}_".upper()
        # 返回带有前缀的环境变量对象
        return self._env.prefixed(prefix)

    def use(self, value: Any | None):
        """
        创建一个上下文管理器，将值推入配置栈。

        Args:
            value: 要推入配置栈的值，可以是任意类型或 None。

        Returns:
            一个上下文管理器对象。
        """

        # 使用 contextmanager 装饰器定义一个上下文管理器
        @contextmanager
        def config_context():
            # 将值推入配置栈，如果值为 None，则使用空字典
            self._config_stack.append(value or {})
            try:
                # yield 语句将控制权交给上下文管理器的调用者
                yield
            finally:
                # finally 块将在上下文管理器退出时执行，确保配置栈被弹出
                self._config_stack.pop()

        # 返回上下文管理器对象
        return config_context()

    @property
    def section(self) -> dict:
        """
        获取当前配置节。

        Returns:
            当前配置节的字典，如果配置栈为空，则返回空字典。
        """
        # 如果配置栈不为空，则返回最后一个元素（即当前配置节）
        # 否则返回空字典
        return self._config_stack[-1] if self._config_stack else {}

    def str(
        self,
        key: KeyValue,
        env_key: EnvKeySet | None = None,
        default_value: str | None = None,
    ) -> str | None:
        """
        读取配置值。

        Args:
            key (KeyValue): 配置键。
            env_key (EnvKeySet | None): 环境变量键。 Defaults to None。
            default_value (str | None): 默认值。 Defaults to None。

        Returns:
            str | None: 配置值或默认值。
        """
        # 读取配置键
        key = read_key(key)

        # 如果当前配置节存在且包含该键，则直接返回配置值
        if self.section and key in self.section:
            return self.section[key]

        # 否则，尝试从环境变量中读取配置值
        return self._read_env(
            # 如果env_key为空，则使用key作为环境变量键
            env_key or key,
            default_value,
            # 使用lambda函数定义读取环境变量的逻辑
            (lambda k, dv: self._env(k, dv))
        )

    def int(
        self,
        key: KeyValue,
        env_key: EnvKeySet | None = None,
        default_value: int | None = None,
    ) -> int | None:
        """
        读取整数配置值。

        Args:
            key (KeyValue): 配置键。
            env_key (EnvKeySet | None): 环境变量键。 Defaults to None。
            default_value (int | None): 默认值。 Defaults to None。

        Returns:
            int | None: 配置值或默认值。
        """
        # 读取配置键
        key = read_key(key)

        # 如果当前配置节存在且包含该键，则直接返回配置值
        if self.section and key in self.section:
            # 将配置值转换为整数
            return int(self.section[key])

        # 否则，尝试从环境变量中读取配置值
        return self._read_env(
            # 如果env_key为空，则使用key作为环境变量键
            env_key or key,
            default_value,
            # 使用lambda函数定义读取环境变量的逻辑
            (lambda k, dv: self._env.int(k, dv))
        )

    def bool(
        self,
        key: KeyValue,
        env_key: EnvKeySet | None = None,
        default_value: bool | None = None,
    ) -> bool | None:
        """
        读取布尔类型的配置值。

        Args:
            key (KeyValue): 配置键。
            env_key (EnvKeySet | None): 环境变量键。 Defaults to None。
            default_value (bool | None): 默认值。 Defaults to None。

        Returns:
            bool | None: 配置值或默认值。
        """
        # 读取配置键
        key = read_key(key)

        # 如果当前配置节存在且包含该键，则直接返回配置值
        if self.section and key in self.section:
            # 将配置值转换为布尔类型
            return bool(self.section[key])

        # 否则，尝试从环境变量中读取配置值
        return self._read_env(
            # 如果env_key为空，则使用key作为环境变量键
            env_key or key,
            default_value,
            # 使用lambda函数定义读取环境变量的逻辑
            lambda k, dv: self._env.bool(k, dv)
        )

    def float(
        self,
        key: KeyValue,
        env_key: EnvKeySet | None = None,
        default_value: float | None = None,
    ) -> float | None:
        """
        读取浮点数配置值。

        Args:
            key (KeyValue): 配置键。
            env_key (EnvKeySet | None): 环境变量键。 Defaults to None。
            default_value (float | None): 默认值。 Defaults to None。

        Returns:
            float | None: 配置值或默认值。
        """
        # 读取配置键
        key = read_key(key)

        # 如果当前配置节存在且包含该键，则直接返回配置值
        if self.section and key in self.section:
            # 将配置值转换为浮点数
            return float(self.section[key])

        # 否则，尝试从环境变量中读取配置值
        return self._read_env(
            # 如果env_key为空，则使用key作为环境变量键
            env_key or key,
            default_value,
            # 使用lambda函数定义读取环境变量的逻辑
            (lambda k, dv: self._env.float(k, dv))
        )

    def list(
        self,
        key: KeyValue,
        env_key: EnvKeySet | None = None,
        default_value: list | None = None,
    ) -> list | None:
        """解析列表配置值。

        参数:
            key (KeyValue): 配置键。
            env_key (EnvKeySet | None): 环境变量键。默认为None。
            default_value (list | None): 默认值。默认为None。

        返回:
            list | None: 配置值或默认值。
        """
        # 读取配置键
        key = read_key(key)

        # 如果当前配置节存在且包含该键，则直接返回配置值
        result = None
        if self.section and key in self.section:
            result = self.section[key]
            if isinstance(result, list):
                return result

        # 否则，尝试从环境变量中读取配置值
        if result is None:
            result = self.str(key, env_key)
        if result is not None:
            # 将配置值按逗号分隔，并去除每个元素的前后空格
            result = [s.strip() for s in result.split(",")]
            # 过滤掉空字符串
            return [s for s in result if s]
        return default_value
