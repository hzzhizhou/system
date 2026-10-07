"""
上下文格式化工具类，用于将检索到的文档格式化为符合要求的字符串。
"""
from pathlib import Path
from typing import List, Dict
from langchain_core.documents import Document
from logs.log_config import generation_layer_log as log

def format_context_with_parents(
    docs: List[Document],
    parent_cache: Dict[str, str],
    max_context_length: int = 4000
) -> str:
    """
    格式化检索到的文档，自动将父子分块的子块替换为父块内容。
    
    Args:
        docs: 检索返回的文档列表（可能包含子块和普通文档）
        parent_cache: 父块映射字典 {parent_id: parent_content}
        max_context_length: 最大上下文长度（字符数），超出则截断
    
    Returns:
        格式化后的上下文字符串
    """
    if not docs:
        return "无参考资料"
    
    parent_scores = {}
    normal_docs = []
    
    for doc in docs:
        # 同一批文档来自同一条检索链路，分数口径一致：开启重排时是 rerank_score，
        # 关闭时是融合分 / 余弦相似度。用 is None 判断，避免 0 分被当成"无分"跳过。
        score = doc.metadata.get("rerank_score")
        if score is None:
            score = doc.metadata.get("fusion_score")
        if score is None:
            score = doc.metadata.get("vector_score", 0.0)
        parent_id = doc.metadata.get("parent_id")
        if parent_id and parent_id in parent_cache:
            if parent_id not in parent_scores:
                parent_scores[parent_id] = {"score": score, "doc": doc}
            else:
                parent_scores[parent_id]["score"] = max(parent_scores[parent_id]["score"], score)
        else:
            normal_docs.append(doc)
    
    sorted_parents = sorted(parent_scores.items(), key=lambda x: x[1]["score"], reverse=True)
    
    context_parts = []
    for parent_id, info in sorted_parents:
        parent_content = parent_cache[parent_id]
        file_name = info["doc"].metadata.get("file_name", "未知文档")
        if "\\" in file_name or "/" in file_name:
            file_name = Path(file_name).name
        context_parts.append(f"来源：{file_name},id:{parent_id},{parent_content}")
    
    for idx, doc in enumerate(normal_docs):
        file_name = doc.metadata.get("file_name", f"未知文档_{idx}")
        if "\\" in file_name or "/" in file_name:
            file_name = Path(file_name).name
        context_parts.append(f"来源：{file_name}】{doc.page_content}")
    
    full_context = "\n\n".join(context_parts)
    if len(full_context) > max_context_length:
        log.warning(f"上下文过长 ({len(full_context)} 字符)，截断至 {max_context_length}")
        full_context = full_context[:max_context_length]
    
    # 可选：保留原有打印调试（可移除或改为日志）
    log.info(f"{full_context[:200]}")
    return full_context