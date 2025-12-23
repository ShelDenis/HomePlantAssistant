from flask import Flask, render_template, request, redirect, url_for
import json
from faiss_manager import FAISSManager
from random import choice

from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import BM25Retriever
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.chat_models import GigaChat
from sentence_transformers import CrossEncoder


API_TOKEN = 'MDE5YTIxNWUtNjlkOC03ZTM5LWE1ZTMtYjk0ZWQ1OGVjYTc5OjMyMjAyMjM5LTY0ZTUtNDVhYy04MGM4LWU2OTMyYzJkNTJhZQ=='


def get_rag_answer_2(question):
    with open("model_creating/data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = [Document(page_content=item["text"]) for item in data]

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 2

    embeddings = HuggingFaceBgeEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    manager = FAISSManager(
                index_dir="cache/faiss_index",
            )
    vectorstore = manager.load_or_create(
                docs=docs,
                embeddings=embeddings,
                data_path="model_creating/data.json"
            )
    faiss_retriever = manager.get_retriever(k=5)

    reranker_model = CrossEncoder(model_name="ai-forever/rugpt3small_based_on_gpt2")


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

        pairs = [[query, doc.page_content] for doc in fused_docs]
        scores = reranker_model.predict(pairs)
        doc_with_scores = list(zip(fused_docs, scores))
        doc_with_scores.sort(key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, score in doc_with_scores[:3]]
        cntxt = "\n\n".join([d.page_content for d in reranked_docs])
        return cntxt

    try:
        llm = GigaChat(temperature=1,
                       verify_ssl_certs=False,
                       credentials=API_TOKEN)
    except Exception as e:
        print(f"Ошибка инициализации GigaChat: {e}")
        exit()

    template = """
    Ответь на вопрос пользователя,
    опираясь *только* на предоставленный ниже контекст.

    Контекст:
    {context}

    Вопрос:
    {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

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

    try:
        response = full_rag_chain.invoke(question)
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


@app.route('/plant_list')
def plant_list_page():
    return render_template('list_page.html')


@app.route('/plant/<plant_slug>')
def plant_detail_page(plant_slug):
    advice_types = ['режим полива',
                    'профилактику от вредителей',
                    'общую информацию',
                    'подходящую освещенность',
                    'возможные заболевания']

    plants_data = {
        'monstera': 'Монстера',
        'ficus': 'Фикус',
        'dracaena': 'Драцена',
        'zamioculcas': 'Замиокулькас',
        'begonia': 'Бегония',
        'fuksia': 'Фуксия',
        'geran': 'Герань',
        'hibiskus': 'Гибискус',
        'kalanchoe': 'Каланхое',
        'mimosa': 'Мимоза',
        'orchideya': 'Орхидея',
        'aloe': 'Алое',
        'aeonium': 'Аэониум',
        'crassula': 'Крассула',
        'echeveria': 'Эхеверия',
        'echinopsys': 'Эхинопсис',
        'hawortia': 'Хавортия',
        'gasteria': 'Гастерия',
        'opuntia': 'Опунция',
    }

    plant = plants_data.get(plant_slug)
    advice = choice(advice_types)

    question = (f'Расскажи про {advice} для растения {plant}')

    answer = get_rag_answer_2(question)

    return render_template('plant_detail_page.html', plant=plant, slug=plant_slug, answer=answer)


if __name__ == 'main':
    app.run(debug=True, timeout=300)
