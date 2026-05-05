"""
Generation configuration for controls like temperature and top-k.
"""

from dataclasses import dataclass, asdict
import json
import os

@dataclass
class GenerationConfig:
    """Settings for text generation."""
    
    max_new_tokens: int = 150
    temperature: float = 0.8
    top_k: int = 40
    do_sample: bool = True
    
    def to_dict(self) -> dict:
        return asdict(self)
        
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "generation_config.json"), "w") as f:
            json.dump(self.to_dict(), f, indent=2)
            
    @classmethod
    def load(cls, path: str):
        file_path = os.path.join(path, "generation_config.json")
        if not os.path.exists(file_path):
            return cls()
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls(**data)
