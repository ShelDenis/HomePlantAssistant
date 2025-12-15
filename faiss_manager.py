"""
Оптимизированный менеджер FAISS с кэшированием и проверками
"""
import os
import json
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from langchain.schema import Document
# from langchain_huggingface import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)


class FAISSManager:
    """
    Менеджер для работы с FAISS индексами с автоматическим кэшированием.
    
    Функции:
    - Сохранение и загрузка индекса с диска
    - Автоматическая проверка изменений в данных
    - Проверка совпадения модели эмбеддингов
    - Добавление новых документов
    - Логирование всех операций
    """
    
    def __init__(
        self,
        index_dir: str = "cache/faiss_index",
        embeddings_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Инициализация менеджера FAISS
        
        Args:
            index_dir: Директория для сохранения индекса
            embeddings_model: Модель для эмбеддингов (от HuggingFace)
        """
        self.index_dir = index_dir
        self.embeddings_model_name = embeddings_model
        self.embeddings = None
        self.vectorstore = None
        
        logger.info(f"FAISSManager инициализирован (индекс: {index_dir})")
    
    # ========================================================================
    # ОСНОВНЫЕ МЕТОДЫ
    # ========================================================================
    
    def load_or_create(
        self,
        docs: List[Document],
        embeddings,
        data_path: Optional[str] = None,
        force_recreate: bool = False
    ) -> FAISS:
        """
        Загрузить FAISS с диска или создать новый индекс.
        
        Алгоритм:
        1. Инициализировать модель эмбеддингов
        2. Проверить, есть ли уже индекс на диске
        3. Проверить, изменились ли данные
        4. Проверить, совпадает ли модель эмбеддингов
        5. Если все проверки пройдены — загрузить из кэша
        6. Иначе — создать и сохранить новый индекс
        
        Args:
            docs: Список документов (нужен для создания индекса)
            data_path: Путь к исходному JSON (для проверки изменений)
            force_recreate: Принудительное пересоздание индекса
        
        Returns:
            FAISS vectorstore
        """
        logger.info("=" * 70)
        logger.info("🚀 Инициализация FAISS")
        logger.info("=" * 70)
        
        # Инициализируем эмбеддинги
        # self._init_embeddings()
        self.embeddings = embeddings
        # Проверка 1: Принудительное пересоздание?
        if force_recreate:
            logger.warning("🔄 FORCE_RECREATE активирован, пересоздаю индекс")
            return self._create_and_save(docs, data_path)
        
        # Проверка 2: Существует ли индекс на диске?
        if not self._is_index_exists():
            logger.info("📝 Индекс не найден, создаю новый...")
            return self._create_and_save(docs, data_path)
        
        # Проверка 3: Изменились ли исходные данные?
        if data_path:
            if self._is_data_changed(data_path):
                logger.warning("⚠️ Исходные данные изменились, пересоздаю индекс...")
                return self._create_and_save(docs, data_path)
        
        # Проверка 4: Совпадает ли модель эмбеддингов?
        # if not self._is_embeddings_model_match():
        #     logger.warning("⚠️ Модель эмбеддингов отличается, пересоздаю индекс...")
        #     return self._create_and_save(docs, data_path)
        
        # Все проверки пройдены — загружаем из кэша
        logger.info("✓ Все проверки пройдены, загружаю индекс из кэша")
        return self._load_from_disk()
    
    def add_documents(self, new_docs: List[Document]) -> None:
        """
        Добавить новые документы к существующему индексу.
        
        Примечание: Это быстрее, чем пересоздание всего индекса.
        
        Args:
            new_docs: Список новых документов для добавления
        """
        if self.vectorstore is None:
            raise RuntimeError(
                "Vectorstore не инициализирован. "
                "Вызовите load_or_create() сначала."
            )
        
        logger.info(f"➕ Добавление {len(new_docs)} новых документов...")
        
        try:
            self.vectorstore.add_documents(new_docs)
            self.vectorstore.save_local(self.index_dir)
            logger.info("✓ Документы добавлены и индекс сохранен")
        except Exception as e:
            logger.error(f"❌ Ошибка при добавлении документов: {e}")
            raise
    
    def get_retriever(self, k: int = 5):
        """
        Получить retriever для поиска по индексу.
        
        Args:
            k: Количество документов для возврата
        
        Returns:
            LangChain retriever
        """
        if self.vectorstore is None:
            raise RuntimeError(
                "Vectorstore не инициализирован. "
                "Вызовите load_or_create() сначала."
            )
        
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
    
    def delete_cache(self) -> None:
        """Удалить кэш индекса с диска."""
        import shutil
        
        if os.path.exists(self.index_dir):
            shutil.rmtree(self.index_dir)
            logger.info(f"🗑️ Кэш удален: {self.index_dir}")
        else:
            logger.warning(f"⚠️ Кэш не найден: {self.index_dir}")
    
    def get_info(self) -> dict:
        """Получить информацию о текущем индексе."""
        metadata_file = os.path.join(self.index_dir, "metadata.json")
        
        if not os.path.exists(metadata_file):
            return {"status": "no_cache"}
        
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            return {
                "status": "cached",
                **metadata
            }
        except Exception as e:
            logger.error(f"Ошибка при чтении метаданных: {e}")
            return {"status": "error", "error": str(e)}
    
    # ========================================================================
    # ВНУТРЕННИЕ МЕТОДЫ (приватные)
    # ========================================================================
    
    # def _init_embeddings(self) -> HuggingFaceBgeEmbeddings:
    #     """Инициализация модели эмбеддингов"""
    #     logger.info(f"🔧 Инициализация embeddings: {self.embeddings_model_name}")
    #     try:
    #         self.embeddings = HuggingFaceBgeEmbeddings(
    #             model_name=self.embeddings_model_name,
    #             device="cpu"  # Измените на "cuda" если есть GPU
    #         )
    #         logger.debug("✓ Embeddings инициализированы")
    #         return self.embeddings
    #     except Exception as e:
    #         logger.error(f"❌ Ошибка инициализации embeddings: {e}")
    #         raise
    
    def _is_index_exists(self) -> bool:
        """Проверка наличия индекса на диске"""
        index_path = Path(self.index_dir)
        index_file = index_path / "index.faiss"
        
        exists = index_path.exists() and index_file.exists()
        
        if exists:
            # Получаем размер индекса
            try:
                size_mb = index_file.stat().st_size / (1024 * 1024)
                logger.info(f"✓ Индекс найден ({size_mb:.1f} MB)")
            except:
                logger.info(f"✓ Индекс найден")
        else:
            logger.info(f"✗ Индекс не найден")
        
        return exists
    
    def _is_data_changed(self, data_path: str) -> bool:
        """
        Проверка изменилась ли исходная база данных.
        
        Используется MD5 хеш для сравнения.
        """
        if not os.path.exists(data_path):
            logger.warning(f"⚠️ Файл данных не найден: {data_path}")
            return True
        
        try:
            # Вычисляем хеш текущих данных
            current_hash = self._compute_file_hash(data_path)
            
            # Проверяем сохраненный хеш
            hash_file = os.path.join(self.index_dir, "data_hash.json")
            
            if os.path.exists(hash_file):
                with open(hash_file, "r") as f:
                    saved_data = json.load(f)
                    saved_hash = saved_data.get("data_hash")
                
                if current_hash == saved_hash:
                    logger.info("✓ Данные не изменились")
                    return False
                else:
                    logger.warning("⚠️ Данные изменились")
                    return True
            else:
                logger.info("ℹ️ Hash файл не найден (первый раз)")
                return True
                
        except Exception as e:
            logger.warning(f"⚠️ Ошибка при проверке хеша: {e}")
            return True
    
    # def _is_embeddings_model_match(self) -> bool:
    #     """Проверка совпадает ли модель эмбеддингов"""
    #     metadata_file = os.path.join(self.index_dir, "metadata.json")
    #
    #     if not os.path.exists(metadata_file):
    #         logger.info("ℹ️ Файл метаданных не найден")
    #         return False
    #
    #     try:
    #         with open(metadata_file, "r") as f:
    #             metadata = json.load(f)
    #
    #         saved_model = metadata.get("embeddings_model")
    #
    #         if saved_model == self.embeddings_model_name:
    #             logger.info(f"✓ Модель совпадает: {saved_model}")
    #             return True
    #         else:
    #             logger.warning(
    #                 f"⚠️ Модель отличается\n"
    #                 f"  Была:   {saved_model}\n"
    #                 f"  Сейчас: {self.embeddings_model_name}"
    #             )
    #             return False
    #     except Exception as e:
    #         logger.warning(f"⚠️ Ошибка при проверке модели: {e}")
    #         return False
    
    def _create_and_save(
        self,
        docs: List[Document],
        data_path: Optional[str] = None
    ) -> FAISS:
        """Создание и сохранение нового индекса"""
        logger.info(f"📝 Создание индекса для {len(docs)} документов...")
        logger.debug("⏳ Это может занять несколько минут...")
        
        try:
            # Создание индекса (долгая операция)
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)
            
            # Создание директории
            os.makedirs(self.index_dir, exist_ok=True)
            
            # Сохранение индекса
            logger.info(f"💾 Сохранение индекса в {self.index_dir}...")
            self.vectorstore.save_local(self.index_dir)
            
            # Сохранение метаданных
            self._save_metadata(docs, data_path)
            
            logger.info("=" * 70)
            logger.info("✓ Индекс успешно создан и сохранен")
            logger.info("=" * 70)
            
            return self.vectorstore
            
        except Exception as e:
            logger.error(f"❌ Ошибка при создании индекса: {e}")
            raise
    
    def _load_from_disk(self) -> FAISS:
        """Загрузка индекса с диска"""
        logger.info(f"⚡ Загрузка индекса с диска...")
        
        try:
            self.vectorstore = FAISS.load_local(
                self.index_dir,
                self.embeddings,
                # allow_dangerous_deserialization=True
            )
            logger.info("=" * 70)
            logger.info("✓ Индекс успешно загружен")
            logger.info("=" * 70)
            return self.vectorstore
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки индекса: {e}")
            raise
    
    def _save_metadata(
        self,
        docs: List[Document],
        data_path: Optional[str] = None
    ) -> None:
        """Сохранение метаданных индекса"""
        metadata = {
            "embeddings_model": self.embeddings_model_name,
            "num_documents": len(docs),
            "created_at": datetime.now().isoformat(),
        }
        
        # Сохраняем хеш исходных данных
        if data_path and os.path.exists(data_path):
            try:
                data_hash = self._compute_file_hash(data_path)
                metadata["data_hash"] = data_hash
                
                with open(os.path.join(self.index_dir, "data_hash.json"), "w") as f:
                    json.dump({"data_hash": data_hash}, f)
                
                logger.debug(f"✓ Data hash сохранен: {data_hash[:8]}...")
            except Exception as e:
                logger.warning(f"⚠️ Не удалось сохранить хеш данных: {e}")
        
        # Сохраняем основные метаданные
        try:
            with open(os.path.join(self.index_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
            logger.debug("✓ Метаданные сохранены")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось сохранить метаданные: {e}")
    
    @staticmethod
    def _compute_file_hash(file_path: str) -> str:
        """Вычисление MD5 хеша файла"""
        md5_hash = hashlib.md5()
        
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    md5_hash.update(chunk)
            return md5_hash.hexdigest()
        except Exception as e:
            logger.error(f"❌ Ошибка при вычислении хеша: {e}")
            raise


# ============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ============================================================================
#
# if __name__ == "__main__":
#     import json
#
#     # Настройка логирования
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     )
#
#     # Функция для загрузки документов из JSON
#     def load_documents_from_json(path: str) -> List[Document]:
#         """Загрузить документы из JSON файла"""
#         logger.info(f"📂 Загрузка документов из {path}")
#
#         with open(path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#
#         docs = [
#             Document(
#                 page_content=item.get("text", ""),
#                 metadata={
#                     "id": idx,
#                     "source": item.get("source", ""),
#                     "title": item.get("title", "")
#                 }
#             )
#             for idx, item in enumerate(data)
#             if "text" in item
#         ]
#
#         logger.info(f"✓ Загружено {len(docs)} документов")
#         return docs
#
#     # ========== СЦЕНАРИЙ 1: Первый запуск ==========
#     print("\n" + "=" * 70)
#     print("СЦЕНАРИЙ 1: Первый запуск (создание индекса)")
#     print("=" * 70)
#
#     # Загружаем документы
#     docs = load_documents_from_json("model_creating/data.json")
#
#     # Создаем менеджер
#     manager = FAISSManager(
#         index_dir="cache/faiss_index",
#         embeddings_model="all-MiniLM-L6-v2"
#     )
#
#     # Загружаем или создаем индекс
#     vectorstore = manager.load_or_create(
#         docs=docs,
#         embeddings=
#         data_path="model_creating/data.json"
#     )
#
#     # Используем retriever
#     retriever = manager.get_retriever(k=3)
#     results = retriever.get_relevant_documents("GStreamer")
#
#     print(f"\n✓ Найдено {len(results)} документов:")
#     for i, doc in enumerate(results, 1):
#         print(f"\n  {i}. {doc.metadata.get('title', 'No title')}")
#         print(f"     {doc.page_content[:100]}...")
#
#     # Информация об индексе
#     print("\n📊 Информация об индексе:")
#     info = manager.get_info()
#     for key, value in info.items():
#         print(f"  {key}: {value}")
#
#     # ========== СЦЕНАРИЙ 2: Повторный запуск ==========
#     print("\n\n" + "=" * 70)
#     print("СЦЕНАРИЙ 2: Повторный запуск (загрузка из кэша)")
#     print("=" * 70)
#
#     # Создаем новый менеджер
#     manager2 = FAISSManager(
#         index_dir="cache/faiss_index",
#         embeddings_model="all-MiniLM-L6-v2"
#     )
#
#     # Загружаем индекс (должно быть быстро)
#     vectorstore2 = manager2.load_or_create(
#         docs=docs,
#         data_path="model_creating/data.json"
#     )
#
#     print("\n✓ Успешно загружен из кэша!")
#
#     # ========== СЦЕНАРИЙ 3: Добавление документов ==========
#     print("\n\n" + "=" * 70)
#     print("СЦЕНАРИЙ 3: Добавление новых документов")
#     print("=" * 70)
#
#     new_docs = [
#         Document(
#             page_content="Новый документ о Kotlin Native",
#             metadata={"id": "new_1", "source": "custom"}
#         )
#     ]
#
#     manager2.add_documents(new_docs)
#     print("✓ Новые документы добавлены")
