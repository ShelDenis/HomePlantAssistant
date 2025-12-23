import os
import json
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from langchain.schema import Document
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)


class FAISSManager:
    def __init__(
        self,
        index_dir: str = "cache/faiss_index",
        embeddings_model: str = "all-MiniLM-L6-v2"
    ):
        self.index_dir = index_dir
        self.embeddings_model_name = embeddings_model
        self.embeddings = None
        self.vectorstore = None

    def load_or_create(
        self,
        docs: List[Document],
        embeddings,
        data_path: Optional[str] = None,
        force_recreate: bool = False
    ) -> FAISS:

        logger.info("=" * 70)
        logger.info("🚀 Инициализация FAISS")
        logger.info("=" * 70)

        self.embeddings = embeddings
        if force_recreate:
            logger.warning("🔄 FORCE_RECREATE активирован, пересоздаю индекс")
            return self._create_and_save(docs, data_path)

        if not self._is_index_exists():
            logger.info("📝 Индекс не найден, создаю новый...")
            return self._create_and_save(docs, data_path)

        if data_path:
            if self._is_data_changed(data_path):
                logger.warning("⚠️ Исходные данные изменились, пересоздаю индекс...")
                return self._create_and_save(docs, data_path)

        logger.info("✓ Все проверки пройдены, загружаю индекс из кэша")
        return self._load_from_disk()
    
    def add_documents(self, new_docs: List[Document]) -> None:

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

        if self.vectorstore is None:
            raise RuntimeError(
                "Vectorstore не инициализирован. "
                "Вызовите load_or_create() сначала."
            )
        
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
    
    def delete_cache(self) -> None:
        import shutil
        
        if os.path.exists(self.index_dir):
            shutil.rmtree(self.index_dir)
            logger.info(f"🗑️ Кэш удален: {self.index_dir}")
        else:
            logger.warning(f"⚠️ Кэш не найден: {self.index_dir}")
    
    def get_info(self) -> dict:
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

    
    def _is_index_exists(self) -> bool:
        index_path = Path(self.index_dir)
        index_file = index_path / "index.faiss"
        
        exists = index_path.exists() and index_file.exists()
        
        if exists:
            try:
                size_mb = index_file.stat().st_size / (1024 * 1024)
                logger.info(f"✓ Индекс найден ({size_mb:.1f} MB)")
            except:
                logger.info(f"✓ Индекс найден")
        else:
            logger.info(f"✗ Индекс не найден")
        
        return exists
    
    def _is_data_changed(self, data_path: str) -> bool:
        if not os.path.exists(data_path):
            logger.warning(f"⚠️ Файл данных не найден: {data_path}")
            return True
        
        try:

            current_hash = self._compute_file_hash(data_path)

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
    
    def _create_and_save(
        self,
        docs: List[Document],
        data_path: Optional[str] = None
    ) -> FAISS:

        logger.info(f"📝 Создание индекса для {len(docs)} документов...")
        logger.debug("⏳ Это может занять несколько минут...")
        
        try:
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)

            os.makedirs(self.index_dir, exist_ok=True)

            logger.info(f"💾 Сохранение индекса в {self.index_dir}...")
            self.vectorstore.save_local(self.index_dir)

            self._save_metadata(docs, data_path)
            
            logger.info("=" * 70)
            logger.info("✓ Индекс успешно создан и сохранен")
            logger.info("=" * 70)
            
            return self.vectorstore
            
        except Exception as e:
            logger.error(f"❌ Ошибка при создании индекса: {e}")
            raise
    
    def _load_from_disk(self) -> FAISS:
        logger.info(f"⚡ Загрузка индекса с диска...")
        
        try:
            self.vectorstore = FAISS.load_local(
                self.index_dir,
                self.embeddings,
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
        metadata = {
            "embeddings_model": self.embeddings_model_name,
            "num_documents": len(docs),
            "created_at": datetime.now().isoformat(),
        }

        if data_path and os.path.exists(data_path):
            try:
                data_hash = self._compute_file_hash(data_path)
                metadata["data_hash"] = data_hash
                
                with open(os.path.join(self.index_dir, "data_hash.json"), "w") as f:
                    json.dump({"data_hash": data_hash}, f)
                
                logger.debug(f"✓ Data hash сохранен: {data_hash[:8]}...")
            except Exception as e:
                logger.warning(f"⚠️ Не удалось сохранить хеш данных: {e}")

        try:
            with open(os.path.join(self.index_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
            logger.debug("✓ Метаданные сохранены")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось сохранить метаданные: {e}")
    
    @staticmethod
    def _compute_file_hash(file_path: str) -> str:
        md5_hash = hashlib.md5()
        
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    md5_hash.update(chunk)
            return md5_hash.hexdigest()
        except Exception as e:
            logger.error(f"❌ Ошибка при вычислении хеша: {e}")
            raise
