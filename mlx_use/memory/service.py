import sqlite3
import logging
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class MemoryService:
    def __init__(self, db_path: Optional[str] = None):
        """Initialize memory service with database"""
        if db_path is None:
            # Default to macOS-use root directory
            db_path = Path(__file__).parent.parent.parent / 'agent_memory.db'

        self.db_path = str(db_path)
        self.model = None  # Lazy load embedding model
        self.initialize_database()

    def initialize_database(self) -> None:
        """Create database and tables if they don't exist"""
        try:
            conn = sqlite3.connect(self.db_path)
            # Enable WAL mode for better concurrency
            conn.execute('PRAGMA journal_mode=WAL')
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    task_context TEXT,
                    embedding BLOB
                )
            ''')

            # Create index for faster queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_source ON memories(source)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at)')

            conn.commit()
            conn.close()
            logger.info(f'Memory database initialized at {self.db_path}')
        except Exception as e:
            logger.error(f'Failed to initialize database: {str(e)}')
            raise

    def save_memory(
        self,
        content: str,
        source: str,
        memory_type: str,
        task_context: Optional[str] = None
    ) -> None:
        """Save a new memory to database"""
        try:
            # Generate embedding for agent memories only
            embedding = None
            if source == 'agent':
                embedding = self._generate_embedding(content)
                # Convert numpy array to bytes for storage
                embedding = embedding.tobytes() if embedding is not None else None

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO memories (content, source, memory_type, task_context, embedding)
                VALUES (?, ?, ?, ?, ?)
            ''', (content, source, memory_type, task_context, embedding))

            conn.commit()
            conn.close()
            logger.debug(f'Saved memory: {content[:50]}... (source={source})')
        except Exception as e:
            logger.error(f'Failed to save memory: {str(e)}')
            # Don't raise - memory failures shouldn't break tasks

    def get_relevant_memories(self, task_description: str) -> List[str]:
        """
        Retrieve relevant memories for current task
        - ALL user-sourced memories
        - Top 5-10 semantically similar agent memories
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get all user memories
            cursor.execute('''
                SELECT content FROM memories
                WHERE source = 'user'
                ORDER BY created_at DESC
            ''')
            user_memories = [row[0] for row in cursor.fetchall()]

            # Get agent memories with embeddings for semantic search
            cursor.execute('''
                SELECT id, content, embedding FROM memories
                WHERE source = 'agent' AND embedding IS NOT NULL
                ORDER BY created_at DESC
                LIMIT 100
            ''')
            agent_rows = cursor.fetchall()

            conn.close()

            # Semantic search for agent memories
            relevant_agent_memories = []
            if agent_rows and task_description:
                task_embedding = self._generate_embedding(task_description)
                if task_embedding is not None:
                    # Calculate similarity scores
                    scores = []
                    for memory_id, content, embedding_bytes in agent_rows:
                        # Convert bytes back to numpy array
                        memory_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)

                        # Cosine similarity
                        similarity = np.dot(task_embedding, memory_embedding) / (
                            np.linalg.norm(task_embedding) * np.linalg.norm(memory_embedding)
                        )
                        scores.append((content, similarity))

                    # Sort by similarity and take top 10
                    scores.sort(key=lambda x: x[1], reverse=True)
                    relevant_agent_memories = [content for content, score in scores[:10] if score > 0.3]

            # Combine: user memories first, then relevant agent memories
            all_memories = user_memories + relevant_agent_memories
            logger.info(f'Loaded {len(user_memories)} user memories and {len(relevant_agent_memories)} agent memories')

            return all_memories

        except Exception as e:
            logger.error(f'Failed to retrieve memories: {str(e)}')
            return []  # Return empty list on error

    def cleanup_old_memories(self) -> int:
        """Delete memories older than 30 days, return count deleted"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cutoff_date = datetime.now() - timedelta(days=30)

            cursor.execute('''
                DELETE FROM memories
                WHERE created_at < ?
            ''', (cutoff_date,))

            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()

            logger.info(f'Cleaned up {deleted_count} old memories')
            return deleted_count

        except Exception as e:
            logger.error(f'Failed to cleanup memories: {str(e)}')
            return 0

    def _generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding vector for text using sentence-transformers"""
        try:
            # Lazy load model
            if self.model is None:
                # Use lightweight model
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.debug('Loaded sentence-transformers model')

            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)

        except Exception as e:
            logger.error(f'Failed to generate embedding: {str(e)}')
            return None
