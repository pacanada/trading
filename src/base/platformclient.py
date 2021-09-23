from abc import ABC, abstractmethod
from dataclasses import dataclass
from src.core.models import BaseModel
from typing import Optional
@dataclass
class PlatformClient(): 
    @abstractmethod
    def get_last_historical_data():
        pass
