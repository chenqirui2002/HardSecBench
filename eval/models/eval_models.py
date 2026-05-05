"""
Data models for eval system
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime


@dataclass
class TestCase:
    """Single test case from result.json"""
    uuid: str
    cwe_id: str
    language: str
    problem_description: str
    module_or_function_name: str
    function_requirements: List[str]
    security_requirements: List[str]
    input_specification: str
    output_specification: str
    golden_file: str
    functional_test_files: Dict[str, str]  # {filename: explanation}
    security_test_files: Dict[str, str]    # {filename: explanation}
    seed_dir: str  # Directory containing test files


@dataclass
class RequirementResult:
    """Result for a single requirement test"""
    requirement_index: int
    requirement_text: str
    requirement_type: str  # 'functional' or 'security'
    tb_file: str
    tb_code: str
    passed: bool
    output: str
    error_message: Optional[str] = None


@dataclass
class IterationResult:
    """Result of a single iteration"""
    iteration_number: int
    code: str
    functional_results: List[RequirementResult] = field(default_factory=list)
    security_results: List[RequirementResult] = field(default_factory=list)
    functional_pass_rate: float = 0.0
    security_pass_rate: float = 0.0
    collaborator_message: Optional[str] = None
    all_functional_passed: bool = False
    code_rolled_back: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class EvalResult:
    """Complete evaluation result for a single test case"""
    uuid: str
    cwe_id: str
    language: str
    model_name: str
    temperature: float
    max_iterations: int
    max_attempts: int
    
    # Pass@k support
    run_index: int = 0  # Index of this run for Pass@k mode (0-based)
    
    # Results
    iterations: List[IterationResult] = field(default_factory=list)
    
    # First iteration pass rates
    first_iter_functional_pass_rate: float = 0.0
    first_iter_security_pass_rate: float = 0.0
    
    # Final iteration pass rates
    final_functional_pass_rate: float = 0.0
    final_security_pass_rate: float = 0.0
    
    total_iterations: int = 0
    success: bool = False  # All functional tests passed
    
    # Timing
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None
    
    # Error info
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "uuid": self.uuid,
            "cwe_id": self.cwe_id,
            "language": self.language,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_iterations": self.max_iterations,
            "max_attempts": self.max_attempts,
            "run_index": self.run_index,
            "iterations": [
                {
                    "iteration_number": it.iteration_number,
                    "functional_pass_rate": it.functional_pass_rate,
                    "security_pass_rate": it.security_pass_rate,
                    "functional_results": [
                        {
                            "requirement_index": r.requirement_index,
                            "requirement_text": r.requirement_text,
                            "passed": r.passed,
                            "error_message": r.error_message
                        } for r in it.functional_results
                    ],
                    "security_results": [
                        {
                            "requirement_index": r.requirement_index,
                            "requirement_text": r.requirement_text,
                            "passed": r.passed,
                            "error_message": r.error_message
                        } for r in it.security_results
                    ],
                    "all_functional_passed": it.all_functional_passed,
                    "code_rolled_back": it.code_rolled_back,
                    "timestamp": it.timestamp
                } for it in self.iterations
            ],
            "first_iter_functional_pass_rate": self.first_iter_functional_pass_rate,
            "first_iter_security_pass_rate": self.first_iter_security_pass_rate,
            "final_functional_pass_rate": self.final_functional_pass_rate,
            "final_security_pass_rate": self.final_security_pass_rate,
            "total_iterations": self.total_iterations,
            "success": self.success,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "error": self.error
        }
