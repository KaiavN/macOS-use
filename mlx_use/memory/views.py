from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel


class MemorySource(str, Enum):
    """Source of memory"""
    USER = 'user'
    AGENT = 'agent'


class MemoryType(str, Enum):
    """Type of memory content"""
    TASK_OUTCOME = 'task_outcome'
    LEARNING = 'learning'
    PREFERENCE = 'preference'
    INSTRUCTION = 'instruction'
    PATTERN = 'pattern'


class Memory(BaseModel):
    """Pydantic model for memory entry"""
    id: Optional[int] = None
    content: str
    source: MemorySource
    memory_type: MemoryType
    created_at: Optional[datetime] = None
    task_context: Optional[str] = None
