"""
Seed generation models for CWE-based problem generation
"""
from typing import List
from pydantic import BaseModel, Field

from models.seed_models import SeedQuestion


class SeedGenerationResult(BaseModel):
    """Result of seed generation for a specific CWE"""
    
    cwe_id: str = Field(..., description="CWE identifier")
    cwe_name: str = Field(..., description="CWE name")
    
    seeds: List[SeedQuestion] = Field(
        default_factory=list,
        description="List of seed questions generated for this CWE"
    )
    
    reasoning: str = Field(..., description="Explanation of seed generation strategy")
    total_seeds: int = Field(
        default=0,
        description="Total number of seeds generated for this CWE"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "cwe_id": "CWE-1191",
                "cwe_name": "On-Chip Debug and Test Interface With Improper Access Control",
                "seeds": [
                    {
                        "question": "Design a debug interface controller with privilege-based access control",
                        "language": "verilog"
                    },
                    {
                        "question": "Implement a JTAG access control mechanism for secure boot",
                        "language": "c"
                    }
                ],
                "reasoning": "Focus on access control scenarios in debug interfaces",
                "total_seeds": 2
            }
        }


class BatchSeedGenerationResult(BaseModel):
    """Batch result of seed generation for multiple CWEs"""
    
    total_cwes: int = Field(..., description="Total number of CWEs processed")
    results: List[SeedGenerationResult] = Field(
        default_factory=list,
        description="Seed generation results for each CWE"
    )
    total_seeds: int = Field(
        default=0,
        description="Total number of seeds across all CWEs"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_cwes": 101,
                "results": [],
                "total_seeds": 850
            }
        }
