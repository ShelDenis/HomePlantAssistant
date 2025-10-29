import chromadb
from chromadb.config import Settings
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import json
from pathlib import Path
import hashlib

class VectorDB:
    """Векторная БД для хранения токенов и документов"""
    
    def __init__(self, embedding_model, persist_dir: str = "./vector_db"):
        # Инициализация ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        print("Загрузка модели эмбеддингов...")
        self.embedding_model = embedding_model

        # Получаем или создаем коллекцию с уникальным именем
        collection_name = f"plants_documents_multilingual"
        
        # Удаляем старую коллекцию если существует (для переиндексации)
        try:
            self.client.delete_collection(collection_name)
            print(f"Старая коллекция {collection_name} удалена для переиндексации")
        except:
            pass
            
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"✓ Векторная БД инициализирована. Документов в базе: {self.collection.count()}")
    
    def load_documents_from_json(self, json_path: str = "model_creating/data.json"):
        """Загрузка документов из существующего JSON файла"""
        
        # Проверяем, не загружены ли уже документы
        if self.collection.count() > 0:
            print(f"ℹ️ БД уже содержит {self.collection.count()} документов.")
            return self.collection.count()
        
        with open(json_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        # Подготавливаем данные для ChromaDB
        ids = []
        texts = []
        metadatas = []
        
        for i, doc in enumerate(documents):
            # Создаем уникальный ID
            doc_id = doc['id'].split('/')[-1].replace('.txt', '')
            if not doc_id:
                doc_id = f"doc_{i}"
            
            # Берем полный текст для лучшего поиска
            full_text = doc['text']
            
            ids.append(doc_id)
            texts.append(full_text)
            metadatas.append({
                'title': doc.get('title', ''),
                'source': doc['id'],
                'doc_index': str(i)
            })
        
        # Создаем эмбеддинги с новой моделью
        print("Создание эмбеддингов с multilingual моделью...")
        embeddings = self.embedding_model.encode(
            texts, 
            show_progress_bar=True,
            normalize_embeddings=True  # Нормализация для лучшего поиска
        )
        
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
        
        # Убеждаемся, что запрос правильно обрабатывается
        query = str(query).strip()
        
        # Создаем эмбеддинг для запроса с нормализацией
        query_embedding = self.embedding_model.encode(
            [query], 
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Поиск в ChromaDB
        try:
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=min(k, self.collection.count())
            )
        except Exception as e:
            print(f"  [ERROR] Ошибка поиска: {e}")
            return []
        
        # Формируем ответ
        documents = []
        if results and results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                doc = {
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else 0
                }
                documents.append(doc)
            
        return documents
    
    def get_stats(self):
        """Статистика БД"""
        return {
            'total_documents': self.collection.count(),
            'collection_name': self.collection.name,
            'embedding_model': self.embedding_model._modules['0'].auto_model.name_or_path
        }