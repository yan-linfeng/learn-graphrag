"""
常用默认配置值
"""

from datashaper import AsyncType

from .enums import (
    CacheType,  # 缓存类型
    InputFileType,  # 输入文件类型
    InputType,  # 输入类型
    LLMType,  # 语言模型类型
    ReportingType,  # 报告类型
    StorageType,  # 存储类型
    TextEmbeddingTarget,  # 文本嵌入目标
)

# 异步模式
ASYNC_MODE = AsyncType.Threaded

# 编码模型
ENCODING_MODEL = "cl100k_base"

# 语言模型类型
# LLM_TYPE = LLMType.OpenAIChat
LLM_TYPE = LLMType.OCIGenAIChat
# 语言模型名称
LLM_MODEL = "gpt-4-turbo-preview"
# 语言模型 ID
LLM_MODEL_ID = "cohere.command-r-plus"
# 最大令牌数
LLM_MAX_TOKENS = 4000
# 温度
LLM_TEMPERATURE = 0
# Top P
LLM_TOP_P = 1
# Top K
LLM_TOP_K = 0
# N
LLM_N = 1
# 请求超时时间
LLM_REQUEST_TIMEOUT = 180.0
# 每分钟令牌数
LLM_TOKENS_PER_MINUTE = 0
# 每分钟请求数
LLM_REQUESTS_PER_MINUTE = 0
# 最大重试次数
LLM_MAX_RETRIES = 10
# 最大重试等待时间
LLM_MAX_RETRY_WAIT = 10.0
# 是否在速率限制推荐时睡眠
LLM_SLEEP_ON_RATE_LIMIT_RECOMMENDATION = True
# 并发请求数
LLM_CONCURRENT_REQUESTS = 25
# 配置文件
LLM_CONFIG_PROFILE = "DEFAULT"

#
# 文本嵌入参数
#
# 文本嵌入类型
# EMBEDDING_TYPE = LLMType.OpenAIEmbedding
EMBEDDING_TYPE = LLMType.OCIGenAIEmbedding
# 文本嵌入模型名称
EMBEDDING_MODEL = "text-embedding-3-small"
# 文本嵌入模型 ID
EMBEDDING_MODEL_ID = "cohere.embed-multilingual-v3.0"
# 批次大小
EMBEDDING_BATCH_SIZE = 16
# 批次最大令牌数
EMBEDDING_BATCH_MAX_TOKENS = 8191
# 文本嵌入目标
EMBEDDING_TARGET = TextEmbeddingTarget.required
# 配置文件
EMBEDDING_CONFIG_PROFILE = "DEFAULT"

# 缓存类型
CACHE_TYPE = CacheType.file
# 缓存基础目录
CACHE_BASE_DIR = "cache"
# 块大小
CHUNK_SIZE = 1200
# 块重叠
CHUNK_OVERLAP = 100
# 块分组列
CHUNK_GROUP_BY_COLUMNS = ["id"]
# 声明描述
CLAIM_DESCRIPTION = (
    "任何与信息发现相关的声明或事实。"
)
# 声明最大获取数
CLAIM_MAX_GLEANINGS = 1
# 声明提取启用
CLAIM_EXTRACTION_ENABLED = False
# 最大聚类大小
MAX_CLUSTER_SIZE = 10
# 社区报告最大长度
COMMUNITY_REPORT_MAX_LENGTH = 2000
# 社区报告最大输入长度
COMMUNITY_REPORT_MAX_INPUT_LENGTH = 8000
# 实体提取实体类型
ENTITY_EXTRACTION_ENTITY_TYPES = ["organization", "person", "geo", "event"]
# 实体提取最大获取数
ENTITY_EXTRACTION_MAX_GLEANINGS = 1
# 输入文件类型
INPUT_FILE_TYPE = InputFileType.text
# 输入类型
INPUT_TYPE = InputType.file
# 输入基础目录
INPUT_BASE_DIR = "input"
# 输入文件编码
INPUT_FILE_ENCODING = "utf-8"
# 输入文本列
INPUT_TEXT_COLUMN = "text"
# 输入 CSV 模式
INPUT_CSV_PATTERN = ".*\\.csv$"
# 输入文本模式
INPUT_TEXT_PATTERN = ".*\\.txt$"
# 并行化延迟
PARALLELIZATION_STAGGER = 0.3
# 并行化线程数
PARALLELIZATION_NUM_THREADS = 50
# Node2Vec 启用
NODE2VEC_ENABLED = False
# Node2Vec 步数
NODE2VEC_NUM_WALKS = 10
# Node2Vec 步长
NODE2VEC_WALK_LENGTH = 40
# Node2Vec 窗口大小
NODE2VEC_WINDOW_SIZE = 2
# 迭代次数
NODE2VEC_ITERATIONS = 3
# 随机种子
NODE2VEC_RANDOM_SEED = 597832
# 报告配置
REPORTING_TYPE = ReportingType.file  # 报告类型
REPORTING_BASE_DIR = "output/${timestamp}/reports"  # 报告基础目录

# 快照配置
SNAPSHOTS_GRAPHML = False  # 是否生成 GraphML 快照
SNAPSHOTS_RAW_ENTITIES = False  # 是否生成原始实体快照
SNAPSHOTS_TOP_LEVEL_NODES = False  # 是否生成顶级节点快照

# 存储配置
STORAGE_BASE_DIR = "output/${timestamp}/artifacts"  # 存储基础目录
STORAGE_TYPE = StorageType.file  # 存储类型

# 摘要描述配置
SUMMARIZE_DESCRIPTIONS_MAX_LENGTH = 500  # 摘要描述最大长度

# UMAP 配置
UMAP_ENABLED = False  # 是否启用 UMAP

# 本地搜索配置
LOCAL_SEARCH_TEXT_UNIT_PROP = 0.5  # 文本单元属性
LOCAL_SEARCH_COMMUNITY_PROP = 0.1  # 社区属性
LOCAL_SEARCH_CONVERSATION_HISTORY_MAX_TURNS = 5  # 对话历史最大回合数
LOCAL_SEARCH_TOP_K_MAPPED_ENTITIES = 10  # 顶K映射实体数
LOCAL_SEARCH_TOP_K_RELATIONSHIPS = 10  # 顶K关系数
LOCAL_SEARCH_MAX_TOKENS = 12_000  # 最大令牌数
LOCAL_SEARCH_LLM_TEMPERATURE = 0  # 语言模型温度
LOCAL_SEARCH_LLM_TOP_P = 1  # 语言模型 Top P
LOCAL_SEARCH_LLM_N = 1  # 语言模型 N
LOCAL_SEARCH_LLM_MAX_TOKENS = 2000  # 语言模型最大令牌数

# 全局搜索配置
GLOBAL_SEARCH_LLM_TEMPERATURE = 0  # 语言模型温度
GLOBAL_SEARCH_LLM_TOP_P = 1  # 语言模型 Top P
GLOBAL_SEARCH_LLM_N = 1  # 语言模型 N
GLOBAL_SEARCH_MAX_TOKENS = 12_000  # 最大令牌数
GLOBAL_SEARCH_DATA_MAX_TOKENS = 12_000  # 数据最大令牌数
GLOBAL_SEARCH_MAP_MAX_TOKENS = 1000  # 映射最大令牌数
GLOBAL_SEARCH_REDUCE_MAX_TOKENS = 2_000  # Reduce 最大令牌数
GLOBAL_SEARCH_CONCURRENCY = 32  # 并发度
