#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""个人科研助手（本地 PDF RAG）

功能：
- 将 ./documents 里的 PDF 建成向量索引（FAISS）
- 命令行交互式问答：对单/多篇文档提问
- 多文档对比分析：输入 compare

依赖：
- Ollama（本地模型服务）
- LangChain + FAISS

示例：
  ollama pull deepseek-r1:8b
  ollama pull qwen3-embedding:0.6b
  python assistant_qwen3.py --docs-folder ./documents --model deepseek-r1:8b --embed-model qwen3-embedding:0.6b
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import argparse

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# RetrievalQA 的导入在不同版本 LangChain 中可能不同，做兼容

from langchain_classic.chains import RetrievalQA


# Ollama LLM 的导入因版本可能不同，做兼容

from langchain_ollama import OllamaLLM


# Ollama Embeddings

from langchain_ollama import OllamaEmbeddings  


class ResearchAssistant:
    def __init__(
        self,
        docs_folder: str = "./documents",
        model_name: str = "deepseek-r1:8b",
        embed_model: str = "qwen3-embedding:0.6b",
        index_dir: str = "./faiss_index",
        rebuild_index: bool = False,
        top_k: int = 4,
    ):
        self.docs_folder = Path(docs_folder)
        self.model_name = model_name
        self.embed_model = embed_model
        self.index_dir = Path(index_dir)
        self.rebuild_index = rebuild_index
        self.top_k = top_k

        self.vectorstore: FAISS | None = None
        self.qa_chain: Any = None
        self.documents_data: Dict[str, Any] = {}

        self.docs_folder.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        print(f"[初始化] 文档文件夹: {self.docs_folder}")
        print(f"[初始化] 生成模型: {self.model_name}")
        print(f"[初始化] Embedding模型: {self.embed_model}")
        print(f"[初始化] 索引目录: {self.index_dir}")

    def load_documents(self) -> List:
        pdf_files = sorted(self.docs_folder.glob("*.pdf"))
        if not pdf_files:
            print(f"警告：在 {self.docs_folder} 中没有找到 PDF 文件")
            return []

        print(f"\n[加载文档] 找到 {len(pdf_files)} 个 PDF 文件")
        all_docs = []
        for pdf in pdf_files:
            print(f"  - 正在加载: {pdf.name}")
            try:
                loader = PyPDFLoader(str(pdf))
                docs = loader.load()
                for d in docs:
                    d.metadata["source_file"] = pdf.name
                all_docs.extend(docs)
                self.documents_data[pdf.name] = docs
                print(f"    ✓ 成功加载 {len(docs)} 页")
            except Exception as e:
                print(f"    ✗ 加载失败: {e}")

        return all_docs

    def _embeddings(self) -> OllamaEmbeddings:
        return OllamaEmbeddings(model=self.embed_model)

    def build_or_load_vectorstore(self, documents: List):
        if not documents:
            raise ValueError("没有文档可以构建向量库")

        index_file = self.index_dir / "index.faiss"
        store_file = self.index_dir / "index.pkl"
        embeddings = self._embeddings()

        if (index_file.exists() and store_file.exists()) and not self.rebuild_index:
            print("\n[加载向量库] 检测到本地索引，正在加载 ...")
            try:
                self.vectorstore = FAISS.load_local(
                    str(self.index_dir),
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
                print("  ✓ 向量库加载完成")
                return
            except Exception as e:
                print(f"  ✗ 加载失败（将改为重建）: {e}")

        print("\n[构建向量库] 正在分langchain_community割文档 ...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = splitter.split_documents(documents)
        print(f"  - 文档分割为 {len(chunks)} 个文本块")

        print("[构建向量库] 正在生成向量并创建 FAISS 索引 ...")
        self.vectorstore = FAISS.from_documents(chunks, embeddings)

        print("[构建向量库] 正在保存索引到本地 ...")
        self.vectorstore.save_local(str(self.index_dir))
        print("  ✓ 向量库构建并保存完成")

    def setup_qa_chain(self):
        if not self.vectorstore:
            raise RuntimeError("向量库未初始化")

        print("\n[初始化QA系统] 正在连接本地 LLM ...")
        llm = OllamaLLM(model=self.model_name, temperature=0.3)

        template = (
            "使用以下检索到的上下文信息来回答问题。如果你不知道答案，请直接说不知道，不要编造答案。\n"
            "请用中文回答，并尽量详细和准确。\n\n"
            "上下文信息：\n{context}\n\n"
            "问题: {question}\n\n"
            "详细回答:"
        )
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": self.top_k}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )
        print("  ✓ QA系统初始化完成")

    def ask(self, question: str) -> Dict[str, Any]:
        if not self.qa_chain:
            return {"error": "QA系统未初始化"}

        print(f"\n[提问] {question}")
        print("[处理中] 正在检索相关文档并生成答案 ...")
        try:
            return self.qa_chain.invoke({"query": question})
        except Exception as e:
            return {"error": f"处理问题时出错: {e}"}

    def compare_documents(self) -> Dict[str, Any] | str:
        if len(self.documents_data) < 2:
            return "需要至少2个文档才能进行比较"

        doc_names = list(self.documents_data.keys())
        q = f"""
我有以下 {len(doc_names)} 篇学术文档：
{', '.join(doc_names)}

请基于这些文档的内容，分析并回答：

1. 这些文档研究的核心问题是什么？相似点和不同点？
2. 这些文档采用了什么研究方法？方法论异同？
3. 这些文档的研究思路/框架有哪些特点？
4. 基于这些文档，推荐值得进一步探索的研究问题。
5. 还有哪些可行的方法或角度？

请尽量引用上下文依据，给出具体、可操作的建议。
""".strip()
        return self.ask(q)

    def interactive(self):
        print("\n" + "=" * 60)
        print("欢迎使用个人科研助手系统")
        print("=" * 60)

        docs = self.load_documents()
        if not docs:
            print("\n请将 PDF 文件放入 documents 文件夹后重新运行程序")
            return

        self.build_or_load_vectorstore(docs)
        self.setup_qa_chain()

        print("\n" + "=" * 60)
        print("系统就绪！您可以开始提问了")
        print("=" * 60)
        print("\n可用命令:")
        print("  - 直接输入问题进行提问")
        print("  - 输入 'compare' 进行多文档比较分析")
        print("  - 输入 'list' 查看已加载的文档")
        print("  - 输入 'quit' 或 'exit' 退出程序")
        print("-" * 60)

        while True:
            try:
                user_input = input("\n您的问题 > ").strip()
                if not user_input:
                    continue

                cmd = user_input.lower()
                if cmd in {"quit", "exit", "q"}:
                    print("\n感谢使用！再见！")
                    break

                if cmd == "list":
                    print("\n已加载的文档:")
                    for i, name in enumerate(self.documents_data.keys(), 1):
                        print(f"  {i}. {name} ({len(self.documents_data[name])} 页)")
                    continue

                if cmd == "compare":
                    print("\n[多文档比较分析]")
                    result = self.compare_documents()
                else:
                    result = self.ask(user_input)

                if isinstance(result, str):
                    print(f"\n{result}")
                    continue

                if "error" in result:
                    print(f"\n错误: {result['error']}")
                    continue

                print(f"\n回答:\n{result.get('result', '无法生成答案')}")

                if "source_documents" in result:
                    sources = {d.metadata.get("source_file", "未知") for d in result["source_documents"]}
                    print(f"\n参考来源 ({len(sources)} 个):")
                    for s in sorted(sources):
                        print(f"  - {s}")

            except KeyboardInterrupt:
                print("\n\n感谢使用！再见！")
                break
            except Exception as e:
                print(f"\n发生错误: {e}")


def main():
    parser = argparse.ArgumentParser(description="个人科研助手 - 本地PDF文档问答系统")
    parser.add_argument("--docs-folder", default="./documents", help="PDF文档所在文件夹 (默认: ./documents)")
    parser.add_argument("--model", default="deepseek-r1:8b", help="Ollama 生成模型名称")
    parser.add_argument(
        "--embed-model",
        default="qwen3-embedding:0.6b",
        help="Ollama Embedding 模型名称 (默认: qwen3-embedding:0.6b)",
    )
    parser.add_argument("--index-dir", default="./faiss_index", help="FAISS 索引保存目录")
    parser.add_argument("--rebuild-index", action="store_true", help="强制重建 FAISS 索引")
    parser.add_argument("--top-k", type=int, default=4, help="检索返回的文本块数量 (默认: 4)")

    args = parser.parse_args()

    assistant = ResearchAssistant(
        docs_folder=args.docs_folder,
        model_name=args.model,
        embed_model=args.embed_model,
        index_dir=args.index_dir,
        rebuild_index=args.rebuild_index,
        top_k=args.top_k,
    )
    assistant.interactive()


if __name__ == "__main__":
    main()
