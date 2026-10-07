"""
融合排序：把多路召回结果合并成单一排序。

RRF 与加权分数是「多路 → 一个排序」这一层的两种策略，二选一（见 FUSION_METHOD），
不串联使用 —— 串联会让先算出的名次被后一步完全覆盖（改造前正是如此：RRF 之后又按
加权分数排了一遍，RRF 白算）。融合之后才是重排（Cross-Encoder，见 retrieval/reranker.py）。
"""
from typing import Dict, List

from langchain_core.documents import Document

from config.settings import BM25_WEIGHT, FUSION_METHOD, RRF_K, VECTOR_WEIGHT
from logs.log_config import retrieval_layer_log as log


def doc_key(doc: Document) -> str:
    """文档去重键：优先业务 id，无 id 时退化为「内容 + 来源」。"""
    if doc.metadata.get("id"):
        return f"id_{doc.metadata['id']}"
    source = doc.metadata.get("source", doc.metadata.get("file_path", ""))
    return f"{doc.page_content}_{source}"


def dedupe_docs(docs: List[Document]) -> List[Document]:
    """按 doc_key 去重，保留首次出现的顺序与对象。"""
    seen = set()
    unique: List[Document] = []
    for doc in docs:
        key = doc_key(doc)
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique


def _collect(bm25_docs: List[Document],
             vector_docs: List[Document]) -> Dict[str, Document]:
    """两路文档合成一张表：同一文档只留一份，并补齐各自带的分数字段。

    BM25 路带 bm25_score、向量路带 vector_score/distance；同一文档命中两路时
    保留先出现的那份对象，只补上它缺的分数字段，供下游展示/门控使用。
    """
    doc_map: Dict[str, Document] = {}
    for docs in (bm25_docs, vector_docs):
        for doc in docs:
            key = doc_key(doc)
            keep = doc_map.setdefault(key, doc)
            if keep is doc:
                continue
            for field in ("bm25_score", "vector_score", "distance"):
                if field in doc.metadata and field not in keep.metadata:
                    keep.metadata[field] = doc.metadata[field]
    return doc_map


def rrf_fuse(bm25_docs: List[Document], vector_docs: List[Document],
             k: int = RRF_K) -> List[Document]:
    """标准 RRF：每路按名次贡献 1/(k+rank)，两路都命中的文档得分累加。

    只用名次、不用分数，天然规避 BM25（无界）与余弦相似度（[0,1]）的量纲差异，
    因此不需要任何权重。
    """
    bm25_docs, vector_docs = bm25_docs or [], vector_docs or []
    doc_map = _collect(bm25_docs, vector_docs)
    scores: Dict[str, float] = {}
    for docs in (bm25_docs, vector_docs):
        for rank, doc in enumerate(docs, start=1):
            key = doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    # sorted 稳定：同分时按插入顺序（BM25 路在前）决定先后
    return [doc_map[key] for key, _ in
            sorted(scores.items(), key=lambda item: item[1], reverse=True)]


def weighted_fuse(bm25_docs: List[Document], vector_docs: List[Document],
                  bm25_weight: float = BM25_WEIGHT,
                  vector_weight: float = VECTOR_WEIGHT) -> List[Document]:
    """加权分数融合：两路分数各自 Min-Max 归一化后按权重相加。

    注意：Min-Max 得到的是批次内相对分 —— 一批全是低相关结果时 top 仍会被拉到 1.0，
    所以只能用于排序，不能用于置信度门控（门控只认绝对分 vector_score）。
    """
    bm25_docs, vector_docs = bm25_docs or [], vector_docs or []
    doc_map = _collect(bm25_docs, vector_docs)
    keys = list(doc_map)
    bm25_norm = _min_max([float(doc_map[key].metadata.get("bm25_score", 0.0)) for key in keys])
    vector_norm = _min_max([float(doc_map[key].metadata.get("vector_score", 0.0)) for key in keys])
    scores = [bm25_weight * b + vector_weight * v
              for b, v in zip(bm25_norm, vector_norm)]
    order = sorted(range(len(keys)), key=lambda i: scores[i], reverse=True)
    return [doc_map[keys[i]] for i in order]


def _min_max(scores: List[float]) -> List[float]:
    """批次内 Min-Max 归一化；全等时返回全 0（避免除零）。"""
    low, high = min(scores), max(scores)
    if high == low:
        return [0.0] * len(scores)
    return [(score - low) / (high - low) for score in scores]


def fuse(bm25_docs: List[Document], vector_docs: List[Document],
         method: str = FUSION_METHOD) -> List[Document]:
    """按 FUSION_METHOD 选择融合策略，返回合并后的候选（不截断）。"""
    if method == "weighted":
        return weighted_fuse(bm25_docs, vector_docs)
    if method != "rrf":
        log.warning(f"未知融合策略 {method!r}，按 rrf 处理")
    return rrf_fuse(bm25_docs, vector_docs)