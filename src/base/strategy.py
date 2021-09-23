from abc import ABC, abstractmethod
from dataclasses import dataclass
from src.core.models import BaseModel
from typing import Optional

@dataclass
class Strategy(ABC):
    take_profit: float
    stop_loss: float
    model: Optional[BaseModel] = None

    @abstractmethod
    def decide():
        pass