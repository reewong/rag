# Neo4j 数据库配置
# NEO4J_URI = "bolt://localhost:7687"  # 替换为您的 Neo4j 服务器 URI
# NEO4J_USER = "neo4j"  # 替换为您的 Neo4j 用户名
# NEO4J_PASSWORD = "password"  # 替换为您的 Neo4j 密码
NEO4J_URI="neo4j+s://fc8667a1.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="liC_71pKt41kOSY1VsS2jDKfrr8kjdg93lLkwsW6f34"

# 项目配置
PROJECT_PATH = "/path/to/cpp/project"  # 替换为您的 C++ 项目实际路径

# LLM 模型配置
LLM_MODEL = "gemma2:27b"  # 当前使用的模型

# 向量存储配置
VECTOR_STORE_PATH = "./vector_store"  # 存储向量数据库的路径

# Doxygen 配置
DOXYGEN_OUTPUT_PATH = "doxygen_output"  # Doxygen 输出相对于项目路径的位置

# 嵌入模型配置
EMBEDDING_MODEL = "unclemusclez/jina-embeddings-v2-base-code"

# 日志配置
LOG_LEVEL = "INFO"
LOG_FILE = "app.log"

# API 配置（如果适用）
API_HOST = "0.0.0.0"
API_PORT = 8000

# 调试配置
DEBUG = False