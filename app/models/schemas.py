from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel


class IntentType(Enum):
    PURCHASE = "purchase"
    COMPARE = "compare"
    SPECS = "specs"
    SUPPORT = "support"
    PRICING = "pricing"
    TECHNICAL = "technical"
    UNKNOWN = "unknown"


@dataclass
class UserProfile:
    gpu_intensity: str
    display_quality: str
    portability: str
    multitasking: str
    processing_speed: str
    budget: float
    primary_use: str
    brand_preference: Optional[str] = None


class UserRequest(BaseModel):
    message: str
    session_id: str


class ChatResponse(BaseModel):
    response: str
    recommendations: Optional[List[Dict]] = None
    similar_products: Optional[List[Dict]] = None
    intent: str
    confidence_score: float = 0.0
