"""知识库 BM25 分数速查：打印真实分块在「文档侧不去重 / 去重」两种分词形态下的打分。

用法（在 RagServerSystem 目录下，用项目 venv）：
    python -m tests.test_tokenizer                  # 默认查询「退货」
    python -m tests.test_tokenizer 退货运费谁承担    # 指定查询词
"""
import sys
import tempfile
from pathlib import Path

from rank_bm25 import BM25Okapi

from utils.tokenizer import tokenizer


def _print_kb_scores(query, top_n=10):
    """用知识库里的真实分块打印 BM25 分数（只读，不写库）。"""
    from infrastructure.vector_store.async_chroma_vector import ChromaVector

    docs = ChromaVector()._get_all_documents()
    if not docs:
        print("知识库为空，跳过。")
        return
    texts = [d.page_content for d in docs]
    names = [d.metadata.get("file_name", "?") for d in docs]

    lines = []

    def out(text=""):
        print(text)
        lines.append(text)

    for label, doc_tokenizer in [
        ("文档侧不去重 —— 现在（tokenize_document）", tokenizer.tokenize_document),
        ("文档侧去重 —— 修复前（tokenize_query 冒充旧 tokenize）", tokenizer.tokenize_query),
    ]:
        toks = [doc_tokenizer(t) for t in texts]
        avgdl = sum(len(t) for t in toks) / len(toks)
        terms = tokenizer.tokenize_query(query)
        scores = BM25Okapi(toks).get_scores(terms)
        order = sorted(range(len(texts)), key=lambda i: scores[i], reverse=True)[:top_n]

        out(f"\n【{label}】  查询={terms}  语料={len(texts)} 块  avgdl={avgdl:.1f}")
        out("-" * 78)
        for rank, i in enumerate(order, 1):
            hits = sum(1 for t in toks[i] if t in terms)
            preview = texts[i][:24].replace("\n", " ")
            out(f"  第{rank:>2}名  分数={scores[i]:>7.4f}  命中={hits:>2}  "
                f"长度={len(toks[i]):>3}  [{names[i]}] {preview}…")

    path = Path(tempfile.gettempdir()) / "kb_bm25_scores.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n完整输出已写入：{path}")


if __name__ == "__main__":
    # 兼容此前带 --kb 的写法：以「--」开头的参数一律忽略
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    _print_kb_scores(args[0] if args else "退货")