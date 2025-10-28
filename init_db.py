from vector_storage import VectorDB
import json
from pathlib import Path


def initialize_vector_db(emb_model):
    """Инициализация векторной БД из существующих файлов"""
    
    # Создаем векторную БД
    db = VectorDB(emb_model, persist_dir="./vector_db")
    
    # Проверяем, есть ли уже документы
    stats = db.get_stats()
    if stats['total_documents'] > 0:
        print(f"⚠️  БД уже содержит {stats['total_documents']} документов")
        response = input("Очистить и загрузить заново? (y/n): ")
        if response.lower() == 'y':
            db.delete_all()
        else:
            return db
    
    # Загружаем из существующего data.json (созданного make_data.ipynb)
    if Path("model_creating/data.json").exists():
        print('Путь БД')
        db.load_documents_from_json("model_creating/data.json")
    else:
        print("❌ Файл data.json не найден. Сначала запустите make_data.ipynb")
        return None
    
    return db

# if __name__ == "__main__":
#     db = initialize_vector_db()
#
#     if db:
#         # Тестовый поиск
#         print("\n🔍 Тестовый поиск:")
#         test_query = "уход за растениями"
#         results = db.search(test_query, k=3)
#
#         for i, result in enumerate(results, 1):
#             print(f"\n{i}. Релевантность: {(1-result['distance'])*100:.1f}%")
#             print(f"   Источник: {result['metadata'].get('source', 'неизвестен')}")
#             print(f"   Текст: {result['text'][:200]}...")
            