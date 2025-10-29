from vector_storage import VectorDB
from pathlib import Path
import sys

def initialize_vector_db(emb_model, force_reload=False):
    """
    Инициализация векторной БД из существующих файлов
    
    Args:
        force_reload: Если True, перезагружает данные даже если БД не пустая
    """
    
    # Создаем векторную БД
    db = VectorDB(emb_model, persist_dir="./vector_db")
    
    # Проверяем, есть ли уже документы
    stats = db.get_stats()
    
    if stats['total_documents'] > 0 and not force_reload:
        print(f"✅ БД уже инициализирована. Документов: {stats['total_documents']}")
        return db
    
    # Если БД пустая или нужна перезагрузка
    if force_reload and stats['total_documents'] > 0:
        print("⚠️ Очистка существующей БД для перезагрузки...")
        # Пересоздаем коллекцию
        db.collection.delete()
        db = VectorDB(emb_model, persist_dir="./vector_db")
    
    # Загружаем из data.json
    if Path("model_creating/data.json").exists():
        print("📄 Загрузка документов из data.json...")
        db.load_documents_from_json("model_creating/data.json")
    else:
        print("❌ Файл data.json не найден. Сначала запустите make_data.ipynb")
        return None
    
    return db

def test_search(db):
    """Тестовая функция для проверки поиска"""
    
    print("\n" + "="*50)
    print("🔍 ТЕСТ ПОИСКА")
    print("="*50)
    
    test_queries = [
        "Листики пожелтели",
        "Кактусы хорошие",
        "Полив растений"
    ]
    
    for query in test_queries:
        print(f"\n📍 ��апрос: '{query}'")
        results = db.search(query, k=2)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. Релевантность: {(1-result['distance'])*100:.1f}%")
            print(f"     Начало текста: {result['text'][:100]}...")

if __name__ == "__main__":
    # Проверяем аргументы командной строки
    force_reload = '--force' in sys.argv
    
    # Инициализируем БД
    db = initialize_vector_db(force_reload=force_reload)
    
    if db:
        # Опционально: тестируем поиск
        if '--test' in sys.argv:
            test_search(db)