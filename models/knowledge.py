# ============================================================
# Knowledge Base for Cultural Context
# ============================================================

import json
import os
from pathlib import Path

class KnowledgeBase:
    """
    Knowledge base for Chinese cultural relics and artifacts.
    Provides historical, cultural, and technical information.
    """
    
    def __init__(self, knowledge_file="data/knowledge_base.json"):
        """
        Initialize knowledge base.
        
        Args:
            knowledge_file (str): Path to knowledge base JSON file
        """
        self.knowledge_file = knowledge_file
        self.data = {}
        
        if os.path.exists(knowledge_file):
            self.load(knowledge_file)
        else:
            self._create_default_kb()
    
    def _create_default_kb(self):
        """
        Create a basic default knowledge base.
        """
        self.data = {
            "materials": {
                "bronze": {
                    "name_en": "Bronze",
                    "name_zh": "青铜",
                    "description": "Bronze alloy used in ancient China for vessels, weapons, and sculptures",
                    "properties": ["durable", "corrosion-resistant", "heavy"],
                    "time_periods": ["Shang Dynasty", "Zhou Dynasty", "Han Dynasty"]
                },
                "stone": {
                    "name_en": "Stone",
                    "name_zh": "石头",
                    "description": "Various stone types including marble, granite used for sculpture",
                    "types": ["marble", "granite", "limestone"],
                    "time_periods": ["All periods"]
                },
                "ceramic": {
                    "name_en": "Ceramic",
                    "name_zh": "陶瓷",
                    "description": "Pottery and porcelain vessels, figurines, and decorative items",
                    "types": ["terracotta", "porcelain", "earthenware"],
                    "time_periods": ["Neolithic", "Shang", "Zhou", "Han", "Tang", "Song"]
                }
            },
            "types": {
                "statue": {
                    "name_en": "Statue",
                    "name_zh": "雕像",
                    "description": "Sculptured representation of a figure",
                    "forms": ["standing", "sitting", "standing"]
                },
                "vessel": {
                    "name_en": "Vessel",
                    "name_zh": "容器",
                    "description": "Container for liquids, food, or ceremonial use",
                    "categories": ["ritual", "domestic", "ceremonial"]
                },
                "painting": {
                    "name_en": "Painting",
                    "name_zh": "绘画",
                    "description": "Artwork created with brush and pigments",
                    "mediums": ["ink", "watercolor", "oil"]
                }
            },
            "dynasties": {
                "shang": {
                    "period": "1600-1046 BCE",
                    "characteristics": ["Bronze vessels", "Oracle bones", "Ritual objects"],
                    "major_artifacts": ["Simuwu Ding", "Bronze vessels", "Jade carvings"]
                },
                "zhou": {
                    "period": "1046-256 BCE",
                    "characteristics": ["Advanced bronze technology", "Ritual bronzes", "Pottery"],
                    "major_artifacts": ["Nine tripods", "Ritual vessels"]
                },
                "han": {
                    "period": "206 BCE - 220 CE",
                    "characteristics": ["Terracotta army", "Jade carvings", "Lacquerware"],
                    "major_artifacts": ["Terracotta Army", "Jade suits"]
                }
            }
        }
        print("✅ Default knowledge base created")
    
    def load(self, knowledge_file):
        """
        Load knowledge base from JSON file.
        
        Args:
            knowledge_file (str): Path to knowledge file
        """
        try:
            with open(knowledge_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"✅ Knowledge base loaded from {knowledge_file}")
        except Exception as e:
            print(f"⚠️  Error loading knowledge base: {e}")
            self._create_default_kb()
    
    def save(self, output_path=None):
        """
        Save knowledge base to JSON file.
        
        Args:
            output_path (str): Path to save knowledge file
        """
        path = output_path or self.knowledge_file
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        print(f"✅ Knowledge base saved to {path}")
    
    def get_material_info(self, material):
        """
        Get information about a material.
        
        Args:
            material (str): Material name
            
        Returns:
            dict: Material information
        """
        material_lower = material.lower()
        for key, info in self.data.get("materials", {}).items():
            if key.lower() == material_lower or info.get("name_en", "").lower() == material_lower:
                return info
        return None
    
    def get_type_info(self, obj_type):
        """
        Get information about an object type.
        
        Args:
            obj_type (str): Object type
            
        Returns:
            dict: Type information
        """
        type_lower = obj_type.lower()
        for key, info in self.data.get("types", {}).items():
            if key.lower() == type_lower or info.get("name_en", "").lower() == type_lower:
                return info
        return None
    
    def get_dynasty_info(self, dynasty):
        """
        Get information about a dynasty.
        
        Args:
            dynasty (str): Dynasty name
            
        Returns:
            dict: Dynasty information
        """
        dynasty_lower = dynasty.lower()
        return self.data.get("dynasties", {}).get(dynasty_lower, None)
    
    def search(self, query, top_k=3):
        """
        Search knowledge base for relevant information.
        
        Args:
            query (str): Search query
            top_k (int): Number of results to return
            
        Returns:
            list: Relevant information snippets
        """
        results = []
        query_lower = query.lower()
        
        # Simple keyword matching
        for category, items in self.data.items():
            if isinstance(items, dict):
                for key, value in items.items():
                    if query_lower in key.lower():
                        results.append({"category": category, "key": key, "data": value})
        
        return results[:top_k]
