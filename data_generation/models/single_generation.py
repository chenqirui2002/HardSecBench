"""
Data models for single CWE generation results
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

from models.generation_models import (
    GenerationTask,
    ProblemDescription, 
    GeneratedCode,
    CodeLanguage
)


class TestCaseResult(BaseModel):
    """Single test case result"""
    
    test_id: str = Field(..., description="Test case ID (e.g., FUNC-TEST-1, SEC-TEST-1)")
    test_name: str = Field(..., description="Test case name")
    test_description: str = Field(..., description="Test case description")
    passed: bool = Field(..., description="Whether the test passed")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    accepted_output: Optional[str] = Field(None, description="Accepted output pattern (finalized, no longer needs LLM judgment)")


class TestSuite(BaseModel):
    """Test suite (functional or security)"""
    
    suite_type: str = Field(..., description="Test type: 'functional' or 'security'")
    test_cases: List[TestCaseResult] = Field(default_factory=list, description="All test cases in this suite")
    total_tests: int = Field(..., description="Total number of test cases")
    passed_tests: int = Field(..., description="Number of passed test cases")
    pass_rate: float = Field(..., description="Pass rate (0.0 to 1.0)")
    test_code: str = Field(..., description="Test code content")
    test_file_path: str = Field(..., description="Path to test file")
    execution_success: bool = Field(..., description="Whether test execution succeeded")
    test_output: Optional[str] = Field(None, description="Test execution output")
    judgments_finalized: bool = Field(False, description="Whether judgments are finalized (uses deterministic rules after finalization, no longer needs LLM)")


class JudgedTestSuite(BaseModel):
    """
    Test suite with LLM judgments (used in per-requirement testing)
    Similar to TestSuite but uses judgment objects instead of test case results
    """
    suite_type: str = Field(..., description="Test type: 'functional' or 'security'")
    test_code: str = Field(..., description="Test code content")
    test_output: str = Field(..., description="Test execution output")
    test_file_path: str = Field(..., description="Path to test file")
    judgments: List[Any] = Field(default_factory=list, description="LLM judgments for each test (TestJudgment objects)")
    total_tests: int = Field(..., description="Total number of tests")
    passed_tests: int = Field(..., description="Number of passed tests")
    failed_tests: int = Field(..., description="Number of failed tests")
    pass_rate: float = Field(..., description="Pass rate (0.0 to 1.0)")


class IterationRecord(BaseModel):
    """Record of one iteration in the generation process"""
    
    iteration_number: int = Field(..., description="Iteration number")
    expert_response: str = Field(..., description="Expert agent's response")
    generated_code: GeneratedCode = Field(..., description="Generated code in this iteration")
    red_teamer_response: Optional[str] = Field(None, description="Red teamer agent's response")
    
    # Detailed test suites
    functional_test_suite: Optional[TestSuite] = Field(None, description="Functional test suite")
    security_test_suite: Optional[TestSuite] = Field(None, description="Security test suite")
    
    vulnerabilities_found: int = Field(0, description="Number of vulnerabilities found")
    vulnerability_details: List[str] = Field(default_factory=list, description="Detailed vulnerability descriptions")
    all_tests_passed: bool = Field(False, description="Whether all tests passed in this iteration")


class SingleGenerationResult(BaseModel):
    """Complete result of a single CWE generation"""
    
    generation_uuid: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this generation")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Generation timestamp")
    
    # Task and seed information
    task: GenerationTask = Field(..., description="Generation task specification")
    
    # Agent outputs
    architect_response: str = Field(..., description="Architect agent's full response")
    problem_description: ProblemDescription = Field(..., description="Generated problem description")
    
    # Iterative generation history
    iterations: List[IterationRecord] = Field(default_factory=list, description="All iterations of the generation process")
    
    # Final outputs
    final_code: GeneratedCode = Field(..., description="Final generated code")
    
    # Test suites
    final_functional_test_suite: Optional[TestSuite] = Field(None, description="Final functional test suite")
    final_security_test_suite: Optional[TestSuite] = Field(None, description="Final security test suite")
    
    # File paths
    code_file_path: str = Field(..., description="Path to the generated code file")
    functional_test_file_path: Optional[str] = Field(None, description="Path to the functional test file")
    security_test_file_path: Optional[str] = Field(None, description="Path to the security test file")
    
    # Summary statistics
    total_iterations: int = Field(..., description="Total number of iterations")
    generation_successful: bool = Field(..., description="Whether generation was successful")
    
    # Detailed test statistics
    total_functional_tests: int = Field(0, description="Total number of functional test cases")
    passed_functional_tests: int = Field(0, description="Number of passed functional test cases")
    functional_pass_rate: float = Field(0.0, description="Functional test pass rate")
    
    total_security_tests: int = Field(0, description="Total number of security test cases")
    passed_security_tests: int = Field(0, description="Number of passed security test cases")
    security_pass_rate: float = Field(0.0, description="Security test pass rate")
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "generation_uuid": "550e8400-e29b-41d4-a716-446655440000",
                "timestamp": "2024-12-06T16:30:00",
                "task": {},
                "total_iterations": 3,
                "generation_successful": True
            }
        }
