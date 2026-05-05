"""
CWE (Common Weakness Enumeration) related models
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class CWEInfo(BaseModel):
    """CWE information from database"""
    
    cwe_id: str = Field(..., description="CWE identifier (e.g., CWE-1191)")
    name: str = Field(..., description="CWE name")
    description: str = Field(..., description="Brief description")
    extended_description: Optional[str] = Field(None, description="Extended description")
    
    abstraction: Optional[str] = Field(default=None, description="CWE abstraction level (Base, Class, etc.)")
    structure: Optional[str] = Field(default=None, description="CWE structure type")
    status: Optional[str] = Field(default=None, description="CWE status (Draft, Stable, etc.)")
    
    applicable_platforms: Optional[str] = Field(default=None, description="Applicable platforms")
    modes_of_introduction: Optional[str] = Field(default=None, description="Phases when weakness is introduced")
    common_consequences: Optional[str] = Field(default=None, description="Common consequences of the weakness")
    potential_mitigations: Optional[str] = Field(default=None, description="Potential mitigation strategies")
    
    demonstrative_examples: Optional[str] = Field(default=None, description="Demonstrative examples")
    observed_examples: Optional[str] = Field(default=None, description="Observed real-world examples")
    related_attack_patterns: Optional[str] = Field(default=None, description="Related CAPEC attack patterns")
    related_weaknesses: Optional[str] = Field(default=None, description="Related CWE weaknesses")
    references: Optional[str] = Field(default=None, description="References and citations")
    notes: Optional[str] = Field(default=None, description="Additional notes")
    
    class Config:
        json_schema_extra = {
            "example": {
                "cwe_id": "CWE-1191",
                "name": "On-Chip Debug and Test Interface With Improper Access Control",
                "description": "The chip does not implement or does not correctly perform access control...",
                "extended_description": "Debug and test interfaces are critical...",
                "applicable_platforms": ["Hardware"],
                "modes_of_introduction": ["Architecture and Design"],
                "common_consequences": ["Confidentiality", "Integrity"]
            }
        }
