from enum import Enum
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from datetime import datetime

class AgentType(str, Enum):
    CODE_ANALYZER = "code_analyzer"
    DEVELOPER = "developer"
    REVIEWER = "reviewer"
    ARCHITECT = "architect"

class AgentState(str, Enum):
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    TERMINATED = "terminated"

class AgentBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    type: AgentType
    description: Optional[str] = None

    @field_validator('name')
    @classmethod
    def name_must_be_valid(cls, v):
        if not v.strip():
            raise ValueError('name cannot be empty or whitespace')
        return v.strip()

class AgentCreate(AgentBase):
    pass

class Agent(AgentBase):
    id: str
    state: AgentState = AgentState.IDLE
    created_at: datetime
    last_active: datetime
    tasks_completed: int = 0

class AgentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    state: Optional[AgentState] = None
