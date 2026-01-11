# app.py - Flask Web 后端
'''
科研助手 Web 版本后端（与 assistant.py 的 ResearchAssistant 对齐）

提供 RESTful API 接口：
- /api/upload      上传 PDF
- /api/documents   获取已加载文档列表
- /api/ask         提问（RAG 问答）
- /api/compare     多文档对比分析（至少 2 篇文档）

说明：
- 后端会在首次提问/对比或上传后自动初始化向量库与 QA 链；
- 上传新文档后会强制重建索引，确保新文档被纳入检索。
'''
from __future__ import annotations

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pathlib import Path
import os
from typing import Any, Dict, List, Tuple, Optional

from assistant import ResearchAssistant


app = Flask(__name__)
CORS(app)

# =========================
# 配置
# =========================
UPLOAD_FOLDER = os.getenv("DOCS_FOLDER", "./documents")
INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "./faiss_index")

ALLOWED_EXTENSIONS = {"pdf"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

# 创建文件夹
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)

# 初始化助手（模型/embedding 可通过环境变量覆盖）
assistant = ResearchAssistant(
    docs_folder=UPLOAD_FOLDER,
    model_name=os.getenv("OLLAMA_MODEL", "deepseek-r1:8b"),
    embed_model=os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b"),
    index_dir=INDEX_DIR,
    rebuild_index=False,
    top_k=int(os.getenv("TOP_K", "4")),
)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _extract_sources(source_documents: Any) -> List[str]:
    """将 source_documents 提炼成更适合前端展示的来源列表"""
    if not source_documents:
        return []

    sources: List[str] = []
    for d in source_documents:
        try:
            meta = getattr(d, "metadata", {}) or {}
            src = meta.get("source_file") or meta.get("source") or "未知来源"
            # PyPDFLoader 通常会给 page；不同版本字段可能不同，做兼容
            page = meta.get("page")
            if page is None:
                page = meta.get("page_number")
            if page is not None:
                sources.append(f"{src} (p.{int(page) + 1})")
            else:
                sources.append(str(src))
        except Exception:
            continue

    # 去重但保持顺序
    seen = set()
    uniq: List[str] = []
    for s in sources:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq


def ensure_ready(force_rebuild: bool = False) -> Tuple[bool, Optional[str]]:
    """确保 documents_data / vectorstore / qa_chain 均已就绪。"""
    # 已就绪则直接返回，避免每次请求都重新加载/构建
    if (not force_rebuild) and assistant.qa_chain and assistant.vectorstore and assistant.documents_data:
        return True, None

    # 同步磁盘上的文档状态（避免 documents_data 残留已删除文件）
    assistant.documents_data.clear()
    docs = assistant.load_documents()
    if not docs:
        return False, "暂无文档，请先上传 PDF。"

    # 强制重建仅在本次调用生效
    prev_rebuild = assistant.rebuild_index
    assistant.rebuild_index = bool(force_rebuild)

    try:
        assistant.build_or_load_vectorstore(docs)
        assistant.setup_qa_chain()
        return True, None
    except Exception as e:
        return False, f"初始化失败: {e}"
    finally:
        assistant.rebuild_index = prev_rebuild


# =========================
# HTML 模板
# =========================
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>个人科研助手</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .message { animation: fadeIn 0.3s ease-in; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    </style>
</head>
<body class="bg-gradient-to-br from-blue-50 to-indigo-50 min-h-screen">
    <div class="container mx-auto px-4 py-8 max-w-6xl">
        <div class="bg-white rounded-2xl shadow-xl overflow-hidden">
            <!-- Header -->
            <div class="bg-gradient-to-r from-indigo-600 to-blue-600 text-white px-8 py-6">
                <h1 class="text-3xl font-bold">🎓 个人科研助手</h1>
                <p class="text-indigo-100 mt-2">上传 PDF 文档，智能分析研究内容</p>
            </div>

            <div class="grid md:grid-cols-3 gap-6 p-8">
                <!-- 左侧：文件上传区 -->
                <div class="md:col-span-1 space-y-4">
                    <div class="bg-gradient-to-br from-indigo-50 to-blue-50 rounded-xl p-6 border-2 border-dashed border-indigo-300">
                        <h3 class="font-semibold text-gray-800 mb-4">📁 上传文档</h3>
                        <input type="file" id="fileInput" multiple accept=".pdf"
                               class="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-indigo-600 file:text-white hover:file:bg-indigo-700 cursor-pointer">
                        <button onclick="uploadFiles()"
                                class="mt-4 w-full bg-indigo-600 text-white py-2 px-4 rounded-lg hover:bg-indigo-700 transition">
                            上传
                        </button>
                        <p class="text-xs text-gray-500 mt-3">提示：上传后会重建索引，确保新文档可被检索。</p>
                    </div>

                    <div class="bg-white rounded-xl p-6 border border-gray-200">
                        <h3 class="font-semibold text-gray-800 mb-4">📚 已加载文档</h3>
                        <div id="fileList" class="space-y-2 text-sm text-gray-600">
                            <p class="text-gray-400">暂无文档</p>
                        </div>
                        <button onclick="reloadDocs()"
                                class="mt-4 w-full bg-gray-100 text-gray-700 py-2 px-4 rounded-lg hover:bg-gray-200 transition text-sm">
                            🔄 刷新列表
                        </button>
                    </div>

                    <div class="bg-amber-50 rounded-xl p-4 border border-amber-200">
                        <h4 class="font-semibold text-amber-800 mb-2 text-sm">💡 快速操作</h4>
                        <div class="space-y-2">
                            <button onclick="compareDocs()"
                                    class="w-full text-left text-sm bg-white px-3 py-2 rounded-lg hover:bg-amber-100 transition">
                                📊 多文档对比分析（compare）
                            </button>
                        </div>
                    </div>
                </div>

                <!-- 右侧：对话区 -->
                <div class="md:col-span-2 flex flex-col h-[600px]">
                    <div id="chatBox" class="flex-1 overflow-y-auto space-y-4 mb-4 p-4 bg-gray-50 rounded-xl">
                        <div class="text-center text-gray-400 py-12">
                            <p class="text-lg">👋 欢迎使用科研助手</p>
                            <p class="text-sm mt-2">上传文档后开始提问</p>
                        </div>
                    </div>

                    <div class="flex gap-3">
                        <input type="text" id="questionInput"
                               placeholder="输入您的问题..."
                               class="flex-1 border border-gray-300 rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                               onkeypress="if(event.key==='Enter') sendQuestion()">
                        <button onclick="sendQuestion()"
                                class="bg-indigo-600 text-white px-6 py-3 rounded-xl hover:bg-indigo-700 transition">
                            发送
                        </button>
                    </div>
                    <div class="text-xs text-gray-500 mt-2">
                        小技巧：也可以在输入框里直接输入 <span class="font-mono bg-white px-1 rounded">compare</span> 来触发多文档对比。
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const API_BASE = '';

        async function uploadFiles() {
            const input = document.getElementById('fileInput');
            const files = input.files;

            if (files.length === 0) {
                alert('请选择文件');
                return;
            }

            const formData = new FormData();
            for (let file of files) {
                formData.append('files', file);
            }

            try {
                addMessage('system', `正在上传 ${files.length} 个文件并重建索引...`);
                const response = await fetch(`${API_BASE}/api/upload`, {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                if (!response.ok) throw new Error(data.error || '上传失败');

                addMessage('system', data.message);
                input.value = '';
                reloadDocs();
            } catch (error) {
                addMessage('system', '上传失败: ' + error.message);
            }
        }

        async function reloadDocs() {
            try {
                const response = await fetch(`${API_BASE}/api/documents`);
                const data = await response.json();
                const fileList = document.getElementById('fileList');

                if (!response.ok) throw new Error(data.error || '获取列表失败');

                if (!data.documents || data.documents.length === 0) {
                    fileList.innerHTML = '<p class="text-gray-400">暂无文档</p>';
                } else {
                    fileList.innerHTML = data.documents.map(doc =>
                        `<div class="bg-indigo-50 px-3 py-2 rounded-lg">📄 ${doc}</div>`
                    ).join('');
                }
            } catch (error) {
                console.error('获取文档列表失败:', error);
            }
        }

        async function compareDocs() {
            addMessage('user', 'compare');
            try {
                const response = await fetch(`${API_BASE}/api/compare`);
                const data = await response.json();
                if (!response.ok) throw new Error(data.error || '对比失败');

                addAssistantAnswer(data);
            } catch (error) {
                addMessage('system', '请求失败: ' + error.message);
            }
        }

        async function sendQuestion() {
            const input = document.getElementById('questionInput');
            const question = input.value.trim();

            if (!question) return;

            addMessage('user', question);
            input.value = '';

            // 输入 compare 也走对比接口
            if (question.toLowerCase() === 'compare') {
                return compareDocs();
            }

            try {
                const response = await fetch(`${API_BASE}/api/ask`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question })
                });
                const data = await response.json();
                if (!response.ok) throw new Error(data.error || '请求失败');

                addAssistantAnswer(data);
            } catch (error) {
                addMessage('system', '请求失败: ' + error.message);
            }
        }

        function addAssistantAnswer(data) {
            // data: { answer, sources } 或 { result, sources }
            const text = data.answer || data.result || data.error || '';
            let content = text;

            if (data.sources && data.sources.length > 0) {
                content += `\n\n—— 参考来源 ——\n` + data.sources.map(s => `• ${s}`).join('\n');
            }
            addMessage('assistant', content);
        }

        function addMessage(role, content) {
            const chatBox = document.getElementById('chatBox');
            if (chatBox.children[0]?.classList.contains('text-center')) {
                chatBox.innerHTML = '';
            }

            const colors = {
                user: 'bg-indigo-600 text-white ml-auto',
                assistant: 'bg-white text-gray-800 shadow-sm',
                system: 'bg-amber-50 text-amber-800 border border-amber-200'
            };

            const messageDiv = document.createElement('div');
            messageDiv.className = `message max-w-3xl rounded-2xl px-5 py-3 ${colors[role]}`;
            messageDiv.innerHTML = `<pre class="whitespace-pre-wrap font-sans text-sm">${content}</pre>`;

            chatBox.appendChild(messageDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        // 初始加载
        reloadDocs();
    </script>
</body>
</html>
"""


# =========================
# 路由
# =========================
@app.route("/")
def index():
    """主页"""
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/upload", methods=["POST"])
def upload_files():
    """上传 PDF 文件（上传后强制重建索引）"""
    if "files" not in request.files:
        return jsonify({"error": "没有文件"}), 400

    files = request.files.getlist("files")
    uploaded: List[str] = []

    for file in files:
        if not file or not file.filename:
            continue
        if not allowed_file(file.filename):
            continue

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        uploaded.append(filename)

    if not uploaded:
        return jsonify({"error": "没有上传有效的 PDF 文件"}), 400

    ok, err = ensure_ready(force_rebuild=True)
    if not ok:
        return jsonify({"error": err}), 500

    return jsonify(
        {
            "message": f"成功上传 {len(uploaded)} 个文件，并已重建索引",
            "files": uploaded,
        }
    )


@app.route("/api/documents", methods=["GET"])
def list_documents():
    """列出已加载的文档"""
    # 懒加载：如果内存里还没加载，就从磁盘同步一次
    if not assistant.documents_data:
        assistant.documents_data.clear()
        assistant.load_documents()

    return jsonify({"documents": list(assistant.documents_data.keys())})


@app.route("/api/ask", methods=["POST"])
def ask_question():
    """回答问题（RAG）"""
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "问题不能为空"}), 400

    ok, err = ensure_ready(force_rebuild=False)
    if not ok:
        return jsonify({"error": err}), 400

    result: Dict[str, Any] = assistant.ask(question)

    if "error" in result:
        return jsonify({"error": result["error"]}), 500

    sources = _extract_sources(result.get("source_documents"))
    return jsonify({"answer": result.get("result", ""), "sources": sources})


@app.route("/api/compare", methods=["GET"])
def compare_documents():
    """多文档比较分析"""
    ok, err = ensure_ready(force_rebuild=False)
    if not ok:
        return jsonify({"error": err}), 400

    result = assistant.compare_documents()

    # compare_documents 可能返回 str 或 dict
    if isinstance(result, str):
        return jsonify({"result": result, "sources": []})

    if isinstance(result, dict) and "error" in result:
        return jsonify({"error": result["error"]}), 500

    sources = _extract_sources(result.get("source_documents") if isinstance(result, dict) else None)
    text = result.get("result", "") if isinstance(result, dict) else str(result)
    return jsonify({"result": text, "sources": sources})


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 科研助手 Web 服务启动")
    print("=" * 60)
    print(f"📁 文档文件夹: {UPLOAD_FOLDER}")
    print(f"🗂️  索引目录: {INDEX_DIR}")
    print("🌐 访问地址: http://localhost:5000")
    print("=" * 60 + "\n")

    app.run(debug=True, host="0.0.0.0", port=5000)
