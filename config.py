# -*- coding: utf-8 -*-
"""
config.py —— RAG 问答机器人统一配置
所有可调参数都放在这里，改这里即可，不用动其他文件。
"""

# ========== 1. 模型配置（本地 Ollama，零成本） ==========
# 主对话模型：生成答案用。安装：ollama pull llama3.1:8b
LLM_MODEL = "llama3.1:8b"
# 嵌入模型：把文本变成向量。安装：ollama pull nomic-embed-text
EMBEDDING_MODEL = "nomic-embed-text"
# Ollama 本地服务地址（默认不动）
OLLAMA_BASE_URL = "http://localhost:11434"

# ========== 2. 向量库配置 ==========
# Chroma 持久化目录，重启程序后知识库不丢
CHROMA_DIR = "./chroma_db"
# 集合名称（知识库名字，可自定义）
COLLECTION_NAME = "company_kb"

# ========== 3. 文档切分参数（重点调参位，影响回答质量） ==========
# 每个文本块的最大字符数：太小上下文断裂，太大幻觉变多，建议 500~1000
CHUNK_SIZE = 800
# 相邻块的重叠字符数：防止一句话被切开导致上下文断裂
CHUNK_OVERLAP = 120

# ========== 4. 检索参数 ==========
# 每次检索返回的文档块数量（TopK），越大信息越多但可能引入噪音
TOP_K = 3

# ========== 5. 安全防护配置 ==========
# 敏感词过滤列表（命中即拒绝回答），可按公司/岗位自定义
SENSITIVE_WORDS = [
    "越狱", "帮我绕过", "忽略所有指令", "ignore previous",
    "忽略系统", "泄露机密", "密码是多少", "后台密码",
]

# 允许的文档类型
ALLOWED_EXTENSIONS = [".pdf", ".docx", ".txt", ".md"]
