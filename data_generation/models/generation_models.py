"""
Data models for code generation tasks
"""
from typing import Optional, List
from pydantic import BaseModel, Field
from enum import Enum


class CodeLanguage(str, Enum):
    """Programming language for code generation"""
    VERILOG = "verilog"
    C = "c"


class GenerationTask(BaseModel):
    """Task specification for code generation"""
    
    task_id: str = Field(..., description="Unique task identifier")
    cwe_id: str = Field(..., description="CWE identifier")
    cwe_name: str = Field(..., description="CWE name")
    cwe_description: str = Field(..., description="CWE description")
    cwe_extended_description: Optional[str] = Field(None, description="Extended CWE description")
    
    # Seed dimensions
    component_id: str = Field(..., description="Component ID (e.g., COMP-001)")
    component_name: str = Field(..., description="Component name (e.g., FSM)")
    component_description: str = Field(..., description="Component description")
    
    trigger_id: str = Field(..., description="Trigger ID (e.g., TRIG-004)")
    trigger_name: str = Field(..., description="Trigger name")
    trigger_description: str = Field(..., description="Trigger description")
    
    # Code naming convention
    language: CodeLanguage = Field(..., description="Target programming language")
    module_name: Optional[str] = Field(None, description="Verilog module name")
    function_name: Optional[str] = Field(None, description="C function name")
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "TASK-CWE1245-COMP001-TRIG004",
                "cwe_id": "CWE-1245",
                "cwe_name": "Improper Finite State Machines (FSMs) in Hardware Logic",
                "cwe_description": "The product uses a FSM that does not properly handle all states...",
                "component_id": "COMP-001",
                "component_name": "FSM (Finite State Machine)",
                "trigger_id": "TRIG-004",
                "trigger_name": "Must Wait N Cycles",
                "language": "verilog",
                "module_name": "vulnerable_fsm"
            }
        }


class ProblemDescription(BaseModel):
    """Generated problem description from Agent A"""
    
    task_id: str = Field(..., description="Task identifier")
    question: str = Field(..., description="Short title + 1-2 sentences describing what this minimal module does")
    function_requirements: List[str] = Field(..., description="2-4 simple functional requirements (just enough to enable the vulnerability)")
    security_requirements: List[str] = Field(
        default_factory=list, 
        description="Empty [] or 1-2 very basic security constraints to verify"
    )
    input_specification: str = Field(..., description="Minimal input ports - typically: clk, rst_n, and 2-4 control/data signals")
    output_specification: str = Field(..., description="Minimal output ports - typically: 1-3 status/data outputs")
    
    # Code naming
    module_or_function_name: str = Field(..., description="Module/function name to implement")
    language: CodeLanguage = Field(..., description="Programming language")
    
    # CWE info (optional)
    cwe_id: Optional[str] = Field(None, description="CWE identifier")
    cwe_name: Optional[str] = Field(None, description="CWE name")


class GeneratedCode(BaseModel):
    """Generated code from Agent B"""
    
    task_id: str = Field(..., description="Task identifier")
    language: CodeLanguage = Field(..., description="Programming language")
    code: str = Field(..., description="Generated code")
    module_or_function_name: str = Field(..., description="Module/function name")
    
    # Validation results
    is_valid: bool = Field(..., description="Whether code is syntactically correct")
    validation_errors: Optional[str] = Field(None, description="Validation error messages")
    compilation_warnings: Optional[str] = Field(None, description="Compilation warnings")


class TestCase(BaseModel):
    """Test case from Agent C"""
    
    task_id: str = Field(..., description="Task identifier")
    language: CodeLanguage = Field(..., description="Programming language")
    test_code: str = Field(..., description="Test code (testbench or test main)")
    
    # Test results
    compilation_success: bool = Field(..., description="Test compilation result")
    execution_success: bool = Field(..., description="Test execution result")
    test_output: Optional[str] = Field(None, description="Test execution output")
    vulnerability_detected: bool = Field(False, description="Whether vulnerability was triggered")
