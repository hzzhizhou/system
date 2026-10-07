from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from infrastructure.EmbeddingService.embedding_service import EmbeddingService
from typing import Any, List, Literal
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import PrivateAttr

from infrastructure.vector_store.async_chroma_vector import ChromaVector
from utils.thread_pool_manager import init_thread_pools
from config.settings import HYBRID_ASYNC_TIMEOUT
from retrieval.bm25_retriever import Bm25Retriever
from retrieval.vector_retriever import VectorRetriever
from retrieval.fusion import fuse
from logs.log_config import retrieval_layer_log as log
import asyncio
import time
class HybridRetriever(BaseRetriever):
    _bm25_retriever: Bm25Retriever = PrivateAttr()
    _vector_retriever: VectorRetriever = PrivateAttr()
    _embedding_service: Any = PrivateAttr()
    def __init__(self, 
                 bm25_retriever: Bm25Retriever, 
                 vector_retriever: VectorRetriever,
                 embedding_service,
                 **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, '_bm25_retriever', bm25_retriever)
        object.__setattr__(self, '_vector_retriever', vector_retriever)
        object.__setattr__(self, '_embedding_service', embedding_service)

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        bm25_docs = self._bm25_retriever._get_relevant_documents(query, **kwargs) or []
        vector_docs = self._vector_retriever._get_relevant_documents(query, **kwargs) or []
        # 合并文档, 并按融合策略排序
        return fuse(bm25_docs, vector_docs)

    async def _aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        bm25_task = None
        vector_task = None
        embedding_task = None
        try:
            start=time.time()
           # 1. 同时启动 BM25 和 embedding 计算           
            bm25_task = asyncio.create_task(
                self._bm25_retriever.ainvoke(query, **kwargs)
            )
            start_time = time.time()
            embedding_task = asyncio.create_task(
                self._embedding_service.embed(query)
            )

            # 2. 等待 embedding 完成，然后启动向量检索
            query_vec = await embedding_task
            end_time = time.time() - start_time
            log.debug(f"embeddings耗时: {end_time}s")
            vector_task = asyncio.create_task(
                self._vector_retriever.search_by_vector(query_vec, **kwargs)
            )
            bm25_docs, vector_docs = await asyncio.wait_for(
                asyncio.gather(bm25_task, vector_task),
                timeout=HYBRID_ASYNC_TIMEOUT
            )
            end =  time.time()-start
            log.info(f"混合检索完成,问题:{query} BM25: {len(bm25_docs)}条, Vector: {len(vector_docs)}条,耗时：{end}")
            return fuse(bm25_docs, vector_docs)
        except asyncio.TimeoutError:
            log.error(f"混合检索超时 (>{HYBRID_ASYNC_TIMEOUT}s) | 查询: {query[:50]}...")
            for task in (bm25_task, vector_task):
                if task and not task.done():
                    task.cancel()
            await asyncio.gather(
                *[t for t in (bm25_task, vector_task) if t and not t.done()],
                return_exceptions=True
            )
            raise
        except Exception as e:
            log.error(f"异步混合检索失败: {e}", exc_info=True)
            raise
