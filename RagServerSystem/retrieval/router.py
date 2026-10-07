import time
from typing import Tuple
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.settings import ROUTE_MODE, LLM_MODEL, LLM_TEMPERATURE, LLM_SEED
from logs.log_config import log

class SmartRouter:
    def __init__(self, bm25_retriever, vector_retriever, hybrid_retriever, llm):
        self.retrievers = {
            "bm25_retriever": bm25_retriever,
            "vector_retriever": vector_retriever,
            "hybrid_retriever": hybrid_retriever
        }
        self.llm = llm
        self.route_chain = self._init_route_chain()

    def _init_route_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是电商售后客服路由助手，根据用户问题返回标签之一：bm25/vector/hybrid
            - bm25:关键词明确的售后查询（如"退货流程"、"运费多少"、"发票怎么开"、"保修期"）
            - vector:语义模糊的咨询/求助（如"我该怎么办"、"为什么还没到"、"能不能换"）
            - hybrid:通用查询（如"这个商品能退吗"、"售后政策"）"""),
            ("human", "问题：{question}")
        ])
        return prompt | self.llm | StrOutputParser()

    async def route(self, question: str, mode: str = ROUTE_MODE) -> Tuple[BaseRetriever, str]:
        try:
            start_time = time.time()
            if mode == "rule":
                # 客服场景关键词：明确售后关键词走混合检索（BM25 保关键词精度 + 向量保语义）
                keyword_patterns = ["退货", "退款", "运费", "发票", "保修", "物流", "订单", "签收", "换货", "维修"]
                # 语义模糊求助走向量检索（语义理解）
                semantic_patterns = ["怎么办", "怎么处理", "为什么", "如何", "能不能", "可以吗", "是不是", "帮我"]
                if any(p in question for p in keyword_patterns):
                    retriever_type = "hybrid_retriever"
                elif any(p in question for p in semantic_patterns):
                    retriever_type = "vector_retriever"
                else:
                    retriever_type = "hybrid_retriever"
            elif mode == "llm":
                result = await self.route_chain.ainvoke({"question": question})
                retriever_type = result.strip().lower()
                if retriever_type not in self.retrievers:
                    retriever_type = "hybrid_retriever"
            else:
                retriever_type = "hybrid_retriever"
            rout_time =  time.time()-start_time
            log.info(f"路由决策：{question[:50]}... → {retriever_type},路由耗时：{rout_time}")
            
            return self.retrievers[retriever_type], retriever_type
        except Exception as e:
            log.error(f"路由失败，降级为混合检索：{e}")
            return self.retrievers["hybrid_retriever"], "hybrid_retriever"