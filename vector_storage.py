import chromadb
from chromadb.config import Settings
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

class VectorDB:
    """Векторная БД для хранения токенов и документов"""
    
    def __init__(self, persist_dir: str = "./vector_db"):
        # Инициализация ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Получаем или создаем коллекцию
        self.collection = self.client.get_or_create_collection(
            name="plants_documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Модель для создания эмбеддингов (та же что в main)
        self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        print(f"✓ Векторная БД инициализирована. Документов в базе: {self.collection.count()}")
    
    def load_documents_from_json(self, json_path: str = "data.json"):
        """Загрузка документов из существующего JSON файла"""
        
        with open(json_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        # Подготавливаем данные для ChromaDB
        ids = []
        texts = []
        metadatas = []
        
        for doc in documents:
            ids.append(doc['id'].split('/')[-1])  # Берем только имя файла
            texts.append(doc['text'])
            metadatas.append({
                'title': doc.get('title', ''),
                'source': doc['id']
            })
        
        # Создаем эмбеддинги
        print("Создание эмбеддингов...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Добавляем в БД
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas
        )
        
        print(f"✓ Загружено {len(documents)} документов в векторную БД")
        return len(documents)
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Поиск релевантных документов"""
        
        # Создаем эмбеддинг запроса
        query_embedding = self.embedding_model.encode([query])
        
        # Поиск в ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=k
        )
        
        # Формируем ответ
        documents = []
        for i in range(len(results['ids'][0])):
            doc = {
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            }
            documents.append(doc)
        
        return documents
    
    def add_document(self, doc_id: str, text: str, metadata: Dict = None):
        """Добавление одного документа"""
        
        embedding = self.embedding_model.encode([text])
        
        self.collection.add(
            ids=[doc_id],
            embeddings=embedding.tolist(),
            documents=[text],
            metadatas=[metadata or {}]
        )
    
    def delete_all(self):
        """Очистка всей коллекции"""
        
        # Получаем все ID
        all_ids = self.collection.get()['ids']
        if all_ids:
            self.collection.delete(ids=all_ids)
            print(f"✓ Удалено {len(all_ids)} документов")
    
    def get_stats(self):
        """Статистика БД"""
        return {
            'total_documents': self.collection.count(),
            'collection_name': self.collection.name
        }
    