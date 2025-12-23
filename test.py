# from flask import Flask, render_template, request, redirect, url_for
# import json
# from tqdm.auto import tqdm
# # from sentence_transformers import SentenceTransformer
# import numpy as np
# import pickle
# # from transformers import AutoTokenizer, AutoModelForCausalLM
# import faiss
# # from sentence_transformers import util
# import re
# from faiss_manager import FAISSManager
#
# from langchain_core.documents import Document
# from langchain_core.runnables import RunnablePassthrough, RunnableLambda
# from langchain_core.output_parsers import StrOutputParser
#
# from langchain.prompts import ChatPromptTemplate
# from langchain.retrievers import BM25Retriever, WikipediaRetriever
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# from langchain.retrievers.document_compressors import FlashrankRerank
# # from langchain_community.cross_encoders import HuggingFaceCrossEncoder
# from langchain_community.chat_models import GigaChat
# from sentence_transformers import CrossEncoder
#
# from init_db import initialize_vector_db
#
#
# question = 'У алое вера начали гнить листья. Хотя корни здоровые. Как быть?'
#
# print('В функцию вошел')
# with open("model_creating/data.json", "r", encoding="utf-8") as f:
#     data = json.load(f)
#
# docs = [Document(page_content=item["text"]) for item in data]
#
# bm25_retriever = BM25Retriever.from_documents(docs)
# bm25_retriever.k = 2
#
# print('bm25_retriever')
#
# embeddings = HuggingFaceBgeEmbeddings(
#     model_name="all-MiniLM-L6-v2"
# )
#
# print('embeddings')
#
# # vectorstore = FAISS.from_documents(docs, embeddings)
# # faiss_retriever = vectorstore.as_retriever()
# manager = FAISSManager(
#     index_dir="cache/faiss_index",
#     # embeddings_model="all-MiniLM-L6-v2"
# )
# vectorstore = manager.load_or_create(
#     docs=docs,
#     embeddings=embeddings,
#     data_path="model_creating/data.json"
# )
# # vectorstore = manager.load_or_create(docs, embeddings, data_path="data.json")
# faiss_retriever = manager.get_retriever(k=5)
#
# print('vectorstore')
#
# print('fuser')
#
# reranker_model = CrossEncoder(model_name="ai-forever/rugpt3small_based_on_gpt2")
#
# print('reranker_model')
#
#
# def hybrid_rrf_retriever(query):
#     bm25_docs = bm25_retriever.get_relevant_documents(query)
#     faiss_docs = faiss_retriever.get_relevant_documents(query)
#
#     fused_scores = {}
#     doc_map = {}
#
#     for rank, doc in enumerate(bm25_docs):
#         key = doc.page_content[:100]
#         fused_scores[key] = fused_scores.get(key, 0) + 1 / (rank + 60)
#         doc_map[key] = doc
#
#     for rank, doc in enumerate(faiss_docs):
#         key = doc.page_content[:100]
#         fused_scores[key] = fused_scores.get(key, 0) + 1 / (rank + 60)
#         if key not in doc_map:
#             doc_map[key] = doc
#
#     sorted_keys = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
#
#     result = [doc_map[key] for key, score in sorted_keys]
#
#     return result
#
#
# def rerank_docs(query, fused_docs):
#     if not fused_docs or len(fused_docs) == 0:
#         return ""
#
#     print('if not fused_docs or len(fused_docs) == 0:')
#
#     pairs = [[query, doc.page_content] for doc in fused_docs]
#     print('pairs')
#     scores = reranker_model.predict(pairs)
#     print('scores')
#     doc_with_scores = list(zip(fused_docs, scores))
#     print('doc_with_scores')
#     doc_with_scores.sort(key=lambda x: x[1], reverse=True)
#     print('doc_with_scores.sort')
#     reranked_docs = [doc for doc, score in doc_with_scores[:3]]
#     print('top-3')
#     cntxt = "\n\n".join([d.page_content for d in reranked_docs])
#     print(cntxt)
#     print('cntxt закончился')
#     return cntxt
#
#
# best_docs = hybrid_rrf_retriever(question)
# reranked_docs = rerank_docs(question, best_docs)
#
# print('best_docs')
# print(best_docs)
# print('reranked_docs')
# print(reranked_docs)
