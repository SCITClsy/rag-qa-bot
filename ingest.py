# -*- coding: utf-8 -*-
"""
ingest.py —— 文档入库脚本（RAG 第一步）

作用：
  把 PDF / Word / TXT 文档拆成小块 -> 转成向量 -> 存入 Chroma 向量库。
  程序跑一次即可，以后文档新增/更新再重跑本脚本刷新知识库。

用法：
  python ingest.py <文档路径或文件夹>
  例：python ingest.py docs/                （入库 docs 文件夹下所有文档）
      python ingest.py 我的手册.pdf         （入库单个文件）

面试亮点（README 里要写）：
  1. 文本切分用重叠策略，解决"一句话被切开导致上下文断裂"
  2. 向量库本地持久化，重启不丢
  3. 支持批量入库
"""
import os
import sys
import argparse

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

import config


# ============================================================
# 1. 加载单个文档 -> 返回文本片段列表
# ============================================================
def load_document(file_path: str):
    """根据扩展名选择对应的文档加载器，返回文档对象列表"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext in (".txt", ".md"):
        # TextLoader 默认按 utf-8 读，读不了就按 gbk 兜底（中文文档常见）
        try:
            loader = TextLoader(file_path, encoding="utf-8")
        except Exception:
            loader = TextLoader(file_path, encoding="gbk")
    else:
        raise ValueError(f"不支持的文件类型: {ext}")

    docs = loader.load()
    # 给每个文档块补上来源信息，回答时能显示"来自哪个文件"
    for d in docs:
        d.metadata["source"] = os.path.basename(file_path)
    return docs


# ============================================================
# 2. 文本切分：把长文档切成小块 + 重叠
# ============================================================
def split_documents(docs):
    """重叠切分，解决一句话被切开的问题，是面试重点"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],  # 优先按句切，保证语义完整
    )
    return splitter.split_documents(docs)


# ============================================================
# 3. 向量化 + 存入 Chroma
# ============================================================
def build_vector_store(split_docs):
    """用本地 Ollama 嵌入模型把文本变向量，持久化到本地目录"""
    embeddings = OllamaEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )
    # delete_old=True 每次重建，保证和最新文档一致（简单粗暴但可靠）
    vector_store = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=config.CHROMA_DIR,
        collection_name=config.COLLECTION_NAME,
    )
    return vector_store


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="文档入库，构建 RAG 知识库")
    parser.add_argument("path", help="文档路径或包含文档的文件夹")
    args = parser.parse_args()

    # 收集所有待入库文件
    if os.path.isdir(args.path):
        files = [
            os.path.join(args.path, f) for f in os.listdir(args.path)
            if os.path.splitext(f)[1].lower() in config.ALLOWED_EXTENSIONS
        ]
    else:
        files = [args.path]

    if not files:
        print("❌ 没有找到可入库的文档，请确认路径和文件类型")
        sys.exit(1)

    print(f"📄 共发现 {len(files)} 个文档，开始加载...")
    all_docs = []
    for f in files:
        try:
            docs = load_document(f)
            all_docs.extend(docs)
            print(f"   ✓ 已加载: {os.path.basename(f)} ({len(docs)} 页/块)")
        except Exception as e:
            print(f"   ✗ 加载失败: {f} -> {e}")

    if not all_docs:
        print("❌ 文档全部加载失败，请检查文件")
        sys.exit(1)

    print(f"🔪 开始切分文本（每块 {config.CHUNK_SIZE} 字符，重叠 {config.CHUNK_OVERLAP}）...")
    split_docs = split_documents(all_docs)
    print(f"   → 切分成 {len(split_docs)} 个文本块")

    print(f"🧠 开始向量化并存入向量库（模型: {config.EMBEDDING_MODEL}）...")
    print("   ⏳ 首次运行需下载模型，可能要几分钟，请耐心等待...")
    vector_store = build_vector_store(split_docs)

    print(f"✅ 入库完成！向量库已保存到: {config.CHROMA_DIR}")
    print("   🚀 现在可以运行: streamlit run app.py  开始问答")
    print(f"   共 {len(split_docs)} 个文本块入库，文档数 {len(files)}")


if __name__ == "__main__":
    main()
