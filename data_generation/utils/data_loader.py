"""
Data loading utilities
"""
import json
from pathlib import Path
from typing import List, Dict, Any
from models.cwe_models import CWEInfo


class DataLoader:
    """Utility class for loading project data"""
    
    @staticmethod
    def load_json(file_path: Path) -> Dict[str, Any]:
        """Load JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def load_cwes(cwe_path: Path) -> List[CWEInfo]:
        """Load CWE database"""
        data = DataLoader.load_json(cwe_path)
        return [CWEInfo(**cwe) for cwe in data]
    
    @staticmethod
    def save_json(data: Any, file_path: Path):
        """Save data to JSON file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            if hasattr(data, 'model_dump'):
                json.dump(data.model_dump(), f, ensure_ascii=False, indent=2)
            elif hasattr(data, '__iter__') and not isinstance(data, (str, dict)):
                json.dump(
                    [item.model_dump() if hasattr(item, 'model_dump') else item for item in data],
                    f, ensure_ascii=False, indent=2
                )
            else:
                json.dump(data, f, ensure_ascii=False, indent=2)
