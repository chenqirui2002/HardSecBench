"""
Seed-related data models
"""
from pydantic import BaseModel, Field


class SeedQuestion(BaseModel):
    """Seed question for a specific CWE scenario"""
    question: str = Field(
        ..., 
        description="1-2 sentence problem description without specific implementation details"
    )
    language: str = Field(
        ..., 
        description="Target implementation language: 'c' or 'verilog'"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "Design a register file with lock bits. The lock mechanism should prevent unauthorized modifications to critical configuration registers.",
                "language": "verilog"
            }
        }
