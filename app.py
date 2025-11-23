from flask import Flask, render_template, request, redirect, url_for
import json
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss
from sentence_transformers import util
import re
import sys

from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import BM25Retriever, WikipediaRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.retrievers.document_compressors import FlashrankRerank
# from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.chat_models import GigaChat
from sentence_transformers import CrossEncoder

from init_db import initialize_vector_db


API_TOKEN = 'MDE5YTIxNWUtNjlkOC03ZTM5LWE1ZTMtYjk0ZWQ1OGVjYTc5OjMyMjAyMjM5LTY0ZTUtNDVhYy04MGM4LWU2OTMyYzJkNTJhZQ=='


def retrieve_relevant_docs(query, model, tokenizer, embedding_model, k=5):
    print('в функцию docs зашел')

    filename = 'model_creating/data.json'

    with open(filename, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    print('json прочитал')

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

    model.config.pad_token_id = tokenizer.pad_token_id

    texts = [d['text'] for d in dataset]

    print('тексты образовал')

    batch_size = 128

    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

    print('батчи сделал')

    document_embeddings = []
    for batch in tqdm(batches):
        batch_embeddings = embedding_model.encode(batch)
        document_embeddings.extend(batch_embeddings)

    document_embeddings = np.array(document_embeddings)

    query_embedding = embedding_model.encode([query])
    print('query_embedding')

    faiss.normalize_L2(document_embeddings)

    dim = document_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(document_embeddings)

    faiss.normalize_L2(query_embedding)
    print('faiss.normalize_L2')
    distances, indices = index.search(query_embedding, k)
    print('Дистанции и индексы нашел')

    relevant_docs = []
    for idx in indices[0]:
        doc = dataset[idx.item()]
        relevant_docs.append(doc)

    return relevant_docs


def extract_paragraphs(text):
    paragraphs = re.split('\n\s*\n', text)
    return list(filter(None, map(str.strip, paragraphs)))


def select_best_paragraph(relevant_docs, query, embedding_model):
    best_paragraphs = []
    for doc in relevant_docs:
        paragraphs = extract_paragraphs(doc['text'])
        # print(paragraphs)
        if len(paragraphs) > 0:
            encoded_query = embedding_model.encode([query])[0]
            encoded_paragraphs = embedding_model.encode(paragraphs)
            similarities = util.cos_sim(encoded_query, encoded_paragraphs)
            best_paragraph_idx = np.argmax(similarities.squeeze())
            best_paragraphs.append(paragraphs[best_paragraph_idx])
    return best_paragraphs


def get_rag_answer(question):
    print('В функцию вошел')
    # with open("model_creating/embopuncia_v1.1.pkl", "rb") as openfile:
    #     embedding_model = pickle.load(openfile)
    # print('Embedding model создал')
    # with open("model_creating/opuncia_v1.1.pkl", "rb") as openfile:
    #     model = pickle.load(openfile)
    # print('Прочитал модель')
    # with open("model_creating/tokopuncia_v1.1.pkl", "rb") as openfile:
    #     tokenizer = pickle.load(openfile)

    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = AutoTokenizer.from_pretrained("ai-forever/rugpt3small_based_on_gpt2")
    model = AutoModelForCausalLM.from_pretrained("ai-forever/rugpt3small_based_on_gpt2")
    # Инициализируем БД
    db = initialize_vector_db(embedding_model)

    print("\n🔍 Тестовый поиск:")
    test_query = question
    results = db.search(test_query, k=3)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. Релевантность: {(1 - result['distance']) * 100:.1f}%")
        print(f"   Источник: {result['metadata'].get('source', 'неизвестен')}")
        print(f"   Текст: {result['text'][:200]}...")

    print('Прочитал модель')
    # relevant_docs = retrieve_relevant_docs(question, model, tokenizer, embedding_model)
    # relevant_docs = [res['text'] for res in results]
    relevant_docs = results
    print('Документы подобрал')
    print(relevant_docs)
    best_sentences = select_best_paragraph(relevant_docs, question, embedding_model)

    print(best_sentences)

    context = ' '.join(list(set(best_sentences)))

    # input_text = f"вопрос: {question}\nконтекст: {context}\n"
    # input_text = f"Вопрос: {question}\nСводки из документов: {context}\nИспользуя указанные сводки, предоставь ответ на вопрос."

    input_text = f"Текст: {context}\n\nВопрос: {question}\nОтвет: "

    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)

    print('Ввод токенизировали')

    outputs = model.generate(inputs.input_ids,
                             max_new_tokens=512,
                             do_sample=True,
                             temperature=0.7,
                             repetition_penalty=1.3)

    print('Вывод получили')

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()[len(input_text) - 1:]


def get_rag_answer_2(question):
    print('В функцию вошел')
    with open("model_creating/data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = [Document(page_content=item["text"]) for item in data]

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 2


    def hybrid_rrf_retriever(query):
        bm25_docs = bm25_retriever.get_relevant_documents(query)
        faiss_docs = faiss_retriever.get_relevant_documents(query)

        fused_scores = {}
        doc_map = {}

        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content[:100]
            fused_scores[key] = fused_scores.get(key, 0) + 1 / (rank + 60)
            doc_map[key] = doc

        for rank, doc in enumerate(faiss_docs):
            key = doc.page_content[:100]
            fused_scores[key] = fused_scores.get(key, 0) + 1 / (rank + 60)
            if key not in doc_map:
                doc_map[key] = doc

        sorted_keys = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        result = [doc_map[key] for key, score in sorted_keys]

        return result

    def rerank_docs(query, fused_docs):
        if not fused_docs or len(fused_docs) == 0:
            return ""

        print('if not fused_docs or len(fused_docs) == 0:')

        pairs = [[query, doc.page_content] for doc in fused_docs]
        print('pairs')
        scores = reranker_model.predict(pairs)
        print('scores')
        doc_with_scores = list(zip(fused_docs, scores))
        print('doc_with_scores')
        doc_with_scores.sort(key=lambda x: x[1], reverse=True)
        print('doc_with_scores.sort')
        reranked_docs = [doc for doc, score in doc_with_scores[:3]]
        print('top-3')
        cntxt = "\n\n".join([d.page_content for d in reranked_docs])
        print(cntxt)
        return cntxt

    print('bm25_retriever')

    embeddings = HuggingFaceBgeEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    print('embeddings')

    vectorstore = FAISS.from_documents(docs, embeddings)
    faiss_retriever = vectorstore.as_retriever()

    print('vectorstore')

    print('fuser')

    reranker_model = CrossEncoder(model_name="ai-forever/rugpt3small_based_on_gpt2")

    print('reranker_model')

    try:
        llm = GigaChat(temperature=1,
                       verify_ssl_certs=False,
                       credentials=API_TOKEN)
        print('GigaChat')
    except Exception as e:
        print(f"Ошибка инициализации GigaChat: {e}")
        exit()

    template = """
    Ты полезный ИИ-ассистент. Отвечай на вопрос пользователя,
    опираясь *только* на предоставленный ниже контекст.

    Контекст:
    {context}

    Вопрос:
    {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    print('prompt')
    full_rag_chain = (
            {
                "fused_docs": RunnableLambda(hybrid_rrf_retriever),
                "question": RunnablePassthrough()
            }
            | RunnableLambda(
        lambda x: {
            "context": rerank_docs(x["question"], x["fused_docs"]),
            "question": x["question"]
        }
    )
            | prompt
            | llm
            | StrOutputParser()
    )
    print('full_rag_chain')
    query = "На алое появились странные пятна. Что делать?"

    try:
        response = full_rag_chain.invoke(question)
        print('response')
        print(response)
        return response
    except Exception as e:
        print(f'Error!: {e}')


app = Flask(__name__)


@app.route('/')
def redirect_to_main():
    return redirect(url_for('main_page'))


@app.route('/main')
def main_page():
    return render_template('main_page.html')


@app.route('/model', methods=['GET', 'POST'])
def rag_answer():
    if request.method == 'POST':
        question = request.form.get("question")
        if not question.strip():
            return render_template('model_page.html', error="Вопрос не задан!")
        try:
            answer = get_rag_answer_2(question)
            return render_template('model_page.html', answer=answer)
        except Exception as e:
            return render_template('model_page.html', error=f"Произошла ошибка: {str(e)}")
    else:
        return render_template('model_page.html')


if __name__ == 'main':
    app.run(debug=True, timeout=300)
