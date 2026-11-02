# -*- coding: utf-8 -*-
"""
app.py —— RAG 问答机器人网页界面（Streamlit）

作用：
  给同事用的可视化问答工具，无需懂代码，上传文档 -> 提问 -> 得到带引用的答案。

启动方式：
  streamlit run app.py
  浏览器会自动打开 http://localhost:8501

功能（面试按这个讲）：
  1. 侧边栏：上传新文档并一键重建知识库（无需命令行）
  2. 聊天问答：检索文档 + 生成答案 + 显示引用来源
  3. 安全防护：敏感词过滤 + Prompt 注入拦截（结合你的测试背景是加分项）
  4. 错误兜底：Ollama 没启动 / 知识库为空时给出友好提示
"""
import os
import tempfile

import streamlit as st
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

import config
import prompts

# ============================================================
# 安全防护模块（你的测试背景亮点，面试重点讲）
# ============================================================
def check_safety(question: str) -> str | None:
    """
    输入安全检查，返回:
      - None 表示通过
      - 字符串表示拦截原因
    拦截两类：敏感词命中 / 疑似 Prompt 注入
    """
    q = question.strip().lower()
    # 1. 敏感词过滤
    for w in config.SENSITIVE_WORDS:
        if w.lower() in q:
            return f"检测到敏感/越权内容（命中: {w}），已拒绝处理。"
    # 2. 简单 Prompt 注入识别
    injection_markers = [
        "忽略上述", "忽略前面", "ignore all previous", "你是", "system prompt",
        "你被越权", "扮演", "忘记你的角色",
    ]
    for m in injection_markers:
        if m.lower() in q:
            return "检测到疑似指令注入，已拦截。请仅提问知识库相关的内容。"
    return None


# ============================================================
# 向量库操作
# ============================================================
@st.cache_resource
def get_vector_store():
    """读取已构建的向量库（只读，不重建）"""
    if not os.path.exists(config.CHROMA_DIR):
        return None
    embeddings = OllamaEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )
    return Chroma(
        persist_directory=config.CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=config.COLLECTION_NAME,
    )


@st.cache_resource
def get_llm():
    """创建对话模型"""
    return ChatOllama(
        model=config.LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.1,  # 低温度，减少随机性，保证答案稳定可靠
    )


def rebuild_knowledge_base(uploaded_files):
    """
    上传新文档并重建知识库。
    复用 ingest.py 里的加载+切分逻辑，保持单一实现。
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from ingest import load_document, build_vector_store

    all_docs = []
    for uf in uploaded_files:
        # 上传的文件是内存对象，先落盘到临时文件再加载
        with tempfile.NamedTemporaryFile(
            suffix=os.path.splitext(uf.name)[1], delete=False
        ) as tmp:
            tmp.write(uf.getbuffer())
            tmp_path = tmp.name
        try:
            docs = load_document(tmp_path)
            all_docs.extend(docs)
        finally:
            os.unlink(tmp_path)  # 清理临时文件

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )
    split_docs = splitter.split_documents(all_docs)
    if not split_docs:
        return 0
    build_vector_store(split_docs)
    return len(split_docs)


# ============================================================
# 回答生成
# ============================================================
def generate_answer(question: str):
    """检索 + 生成，返回 (答案, 引用来源列表)"""
    vector_store = get_vector_store()
    llm = get_llm()

    # 1. 检索最相关的文本块
    retriever = vector_store.as_retriever(search_kwargs={"k": config.TOP_K})
    retrieved_docs = retriever.invoke(question)

    # 2. 组装上下文 + 提示词
    context_parts = []
    sources = []
    for i, doc in enumerate(retrieved_docs, 1):
        context_parts.append(f"[片段{i}]\n{doc.page_content}")
        src = doc.metadata.get("source", "未知来源")
        if src not in sources:
            sources.append(src)

    context = "\n\n".join(context_parts)
    user_prompt = prompts.build_user_prompt(question, context)

    # 3. 调用大模型生成
    response = llm.invoke(
        [
            ("system", prompts.rag_system_prompt()),
            ("human", user_prompt),
        ]
    )
    return response.content, sources


# ============================================================
# Streamlit 界面
# ============================================================
st.set_page_config(page_title="📚 公司知识库问答助手", page_icon="📚", layout="wide")

st.title("📚 公司知识库问答助手")
st.caption("基于 RAG：上传文档 → 自动建库 → 基于资料回答，附引用来源，杜绝编造")

# ---------- 侧边栏：知识库管理 ----------
with st.sidebar:
    st.header("🗂 知识库管理")
    uploaded_files = st.file_uploader(
        "上传文档（PDF/Word/TXT，可多选）",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        if st.button("🚀 入库并重建知识库", use_container_width=True):
            with st.spinner("正在切分+向量化，请稍候..."):
                n = rebuild_knowledge_base(uploaded_files)
            if n:
                st.success(f"✅ 入库完成，共 {n} 个文本块")
                # 知识库变了，清掉缓存
                get_vector_store.clear()
            else:
                st.error("入库失败，请检查文件")
    else:
        st.info("📌 提示：也可以用命令行 python ingest.py docs/ 批量入库")

    st.divider()
    st.header("⚙️ 当前配置")
    st.write(f"- 对话模型：{config.LLM_MODEL}")
    st.write(f"- 嵌入模型：{config.EMBEDDING_MODEL}")
    st.write(f"- 检索返回：Top {config.TOP_K} 片段")

# ---------- 主区域：聊天问答 ----------
vector_store = get_vector_store()
if vector_store is None:
    st.warning(
        "⚠️ 知识库为空。请先在左侧上传文档并点击入库，"
        "或命令行运行 `python ingest.py docs/`。\n\n"
        "另外请确认 Ollama 已启动：`ollama serve`，并已拉取模型。"
    )
    st.stop()

# 初始化会话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            st.caption("📎 引用来源：" + "、".join(msg["sources"]))

# 输入框
if prompt := st.chat_input("请输入你的问题，例如：请假制度是什么？"):
    # 安全校验（结合你的测试背景）
    safety_block = check_safety(prompt)
    if safety_block:
        with st.chat_message("assistant"):
            st.error(safety_block)
            st.session_state.messages.append(
                {"role": "assistant", "content": f"🚫 {safety_block}"}
            )
        st.stop()

    # 用户问题入历史
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 生成回答
    with st.chat_message("assistant"):
        with st.spinner("正在检索知识库并生成回答..."):
            try:
                answer, sources = generate_answer(prompt)
            except Exception as e:
                answer = f"❌ 出错了：{e}\n\n请确认 Ollama 已启动（`ollama serve`）并已拉取模型。"
                sources = []
        st.markdown(answer)
        if sources:
            st.caption("📎 引用来源：" + "、".join(sources))
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
